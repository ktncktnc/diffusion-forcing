from functools import partial
from typing import Optional
import torch
from torch import nn
from einops import rearrange
from omegaconf import DictConfig
from rotary_embedding_torch import RotaryEmbedding
from .embeddings import StochasticTimeEmbedding, ZtoVDTConvAdapter
from .u_net_blocks import *


class IdentityWithExtraArgs(nn.Identity):
    def forward(self, x, *args, **kwargs):
        return x



class Unet3D(nn.Module):
    def __init__(
        self,
        cfg: DictConfig,
        input_size: torch.Size, # h
        in_channels: int, # c
        z_channels: int, # condition c,
        # init_hidden_channels: int = 32,
        # dim_mults: list = [1, 2, 4, 8],
        # num_res_blocks: int = 2,
        # resnet_block_groups: int = 4,
        # attn_resolutions: list = [8],
        # attn_heads: int = 1,
        # attn_head_dim: int = 32,
        # use_linear_attn: bool = False,
        # use_init_temporal_attn: bool = True,
        # init_kernel_size: int = 1,
        # dropout: float = 0.0,
        # is_causal_selfattn: bool = True,
        # is_causal_crossattn: bool = True,
        # use_fourier_noise_embedding: bool = False,
        # use_fourier_cond_embedding: bool = False,
    ):
        super().__init__()

        # self.cond_channel = cfg.z_shape[0]
        self.cond_channel = z_channels
        self.is_causal_selfattn = cfg.is_causal_selfattn
        self.is_causal_crossattn = cfg.is_causal_crossattn
        self.use_fourier_noise_embedding = cfg.use_fourier_noise_embedding
        self.use_fourier_cond_embedding = cfg.use_fourier_cond_embedding
        self.use_init_temporal_attn = cfg.use_init_temporal_attn
        self.use_linear_attn = cfg.use_linear_attn
        self.attn_resolutions = cfg.attn_resolutions

        self.dim_mults = cfg.dim_mults
        self.init_hidden_channels = cfg.init_hidden_channels
        self.init_kernel_size = cfg.init_kernel_size

        self.num_heads = cfg.num_heads
        self.attn_head_dim = cfg.attn_head_dim

        self.num_res_blocks = cfg.num_res_blocks
        self.resnet_block_groups = cfg.resnet_block_groups

        resolution = input_size
        out_channel = in_channels
        self.attn_resolutions = [resolution // res for res in list(self.attn_resolutions)]

        dims = [self.init_hidden_channels, *map(lambda m: self.init_hidden_channels * m, self.dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        mid_dim = dims[-1]

        emb_dim = self.noise_level_emb_dim

        init_padding = self.init_kernel_size // 2       
        self.init_conv = nn.Conv3d(
            in_channels,
            self.init_hidden_channels,
            kernel_size=(1, self.init_kernel_size, self.init_kernel_size),
            padding=(0, init_padding, init_padding),
        )

        self.noise_level_pos_embedding = StochasticTimeEmbedding(
            dim=self.noise_level_dim,
            time_embed_dim=self.noise_level_emb_dim,
            use_fourier=self.use_fourier_noise_embedding,
        )

        self.cond_embedding = ZtoVDTConvAdapter(
            self.cond_channel,
            self.noise_level_emb_dim,
            use_fourier=self.use_fourier_cond_embedding
        )

        self.rotary_time_pos_embedding = RotaryEmbedding(dim=self.attn_head_dim)

        self.init_temporal_attn = (
            UnetTemporalAttentionBlock(
                dim=self.init_hidden_channels,
                heads=self.num_heads,
                dim_head=self.attn_head_dim,
                is_causal=self.is_causal_selfattn,
                rotary_emb=self.rotary_time_pos_embedding,
            )
            if self.use_init_temporal_attn
            else IdentityWithExtraArgs()
        )

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        block_klass = partial(ResnetBlock, groups=self.resnet_block_groups)
        block_klass_noise = partial(
            ResnetBlock, groups=self.resnet_block_groups, emb_dim=emb_dim
        )
        spatial_attn_klass = partial(
            UnetSpatialAttentionBlock, heads=self.num_heads, dim_head=self.attn_head_dim
        )

        temporal_attn_klass = partial(
            UnetTemporalAttentionBlock,
            heads=self.num_heads,
            dim_head=self.attn_head_dim,
            is_causal=self.is_causal_crossattn, #TODO:d: changed from crossattn to selfattn
            rotary_emb=self.rotary_time_pos_embedding,
        )
        
        crossattn_klass = partial(
            UnetTemporalCrossAttentionBlock,
            heads=self.num_heads,
            dim_head=self.attn_head_dim,
            is_causal=self.is_causal_crossattn, #TODO:
            rotary_emb=self.rotary_time_pos_embedding,
        )

        curr_resolution = 1
        for idx, (dim_in, dim_out) in enumerate(in_out):
            is_last = idx == len(in_out) - 1
            use_attn = curr_resolution in self.attn_resolutions

            self.down_blocks.append(
                nn.ModuleList(
                    [
                        UnetSequentialCondition(
                            block_klass_noise(dim_in, dim_out),
                            *(
                                block_klass_noise(dim_out, dim_out)
                                for _ in range(self.num_res_blocks - 1)
                            ),
                            (
                                spatial_attn_klass(
                                    dim_out,
                                    use_linear=self.use_linear_attn and not is_last,
                                )
                                if use_attn
                                else nn.Identity()
                            ),
                            temporal_attn_klass(dim_out) if use_attn else nn.Identity(),
                            crossattn_klass(
                                query_dim=dim_out,
                                key_dim=self.noise_level_emb_dim,
                                value_dim=self.noise_level_emb_dim
                            ),
                        ),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

            curr_resolution *= 2 if not is_last else 1

        self.mid_block = UnetSequentialCondition(
            block_klass_noise(mid_dim, mid_dim),
            spatial_attn_klass(mid_dim),
            temporal_attn_klass(mid_dim),
            crossattn_klass(
                query_dim=dim_out,
                key_dim=self.noise_level_emb_dim,
                value_dim=self.noise_level_emb_dim
            ),
            block_klass_noise(mid_dim, mid_dim),
        )

        for idx, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = idx == len(in_out) - 1
            use_attn = curr_resolution in self.attn_resolutions

            self.up_blocks.append(
                UnetSequentialCondition(
                    block_klass_noise(dim_out * 2, dim_in),
                    *(
                        block_klass_noise(dim_in, dim_in)
                        for _ in range(self.num_res_blocks - 1)
                    ),
                    (
                        spatial_attn_klass(
                            dim_in, use_linear=self.use_linear_attn and idx > 0
                        )
                        if use_attn
                        else nn.Identity()
                    ),
                    temporal_attn_klass(dim_in) if use_attn else nn.Identity(),
                    crossattn_klass(
                        query_dim=dim_in,
                        key_dim=self.noise_level_emb_dim,
                        value_dim=self.noise_level_emb_dim
                    ),
                    Upsample(dim_in) if not is_last else nn.Identity(),
                )
            )
            curr_resolution //= 2 if not is_last else 1

        self.out = nn.Sequential(block_klass(self.init_hidden_channels*2, self.init_hidden_channels), nn.Conv3d(self.init_hidden_channels, out_channel, 1))

    @property
    def noise_level_emb_dim(self):
        return self.init_hidden_channels * 4
    
    @property
    def noise_level_dim(self):
        return max(self.noise_level_emb_dim // 4, 32)

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor],
        t: torch.Tensor,
        external_cond_mask: Optional[torch.Tensor] = None,
    ):
        x = rearrange(x, "b t c h w -> b c t h w").contiguous()

        # TODO: patchify x

        difffusion_time_emb = self.noise_level_pos_embedding(t) # (b, t, emb_dim)
        cond_emb = None
        if self.cond_embedding is not None:
            if cond is None:
                raise ValueError("External condition is required, but not provided.")
            cond_emb = self.cond_embedding(
                cond
            ) # (b, t, emb_dim)

        x = self.init_conv(x) # (b, init_dim, t, h, w)

        # It adds time embedding here, self attn
        x = self.init_temporal_attn(x)
        h = x.clone()

        hs = []

        for block, downsample in self.down_blocks:
            h = block(h, cond_emb, difffusion_time_emb)
            hs.append(h)
            h = downsample(h)

        h = self.mid_block(h, cond_emb, difffusion_time_emb)
        for block in self.up_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = block(h, cond_emb, difffusion_time_emb)

        h = torch.cat([h, x], dim=1)
        x = self.out(h)
        x = rearrange(x, "b c t h w -> b t c h w")
        return x


# import torch
# import matplotlib.pyplot as plt
# from omegaconf import OmegaConf
# from algorithms.rnn_diffusion_correction.models.u_net3d import Unet3D
# from algorithms.rnn_diffusion_correction.models.u_net_blocks import UnetTemporalAttentionBlock, UnetTemporalCrossAttentionBlock

# def disable_norm_layers(model):
#     """Replace all normalization layers with identity"""
#     for module in model.modules():
#         if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
#                               torch.nn.LayerNorm, torch.nn.GroupNorm, torch.nn.InstanceNorm1d,
#                               torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d)):
#             # Save the original state and replace with identity
#             module._original_forward = module.forward
#             module.forward = lambda x: x
#     return model

# def test_component(module_name, module, input_shape, is_cross=False, cond_shape=None):
#     """Test a single component for causal behavior"""
#     print(f"\nTesting {module_name}...")
    
#     # Create inputs
#     torch.manual_seed(42)
#     x1 = torch.randn(*input_shape)
#     x2 = x1.clone()
    
#     # Change future timesteps in x2
#     time_dim = 2 if len(input_shape) > 3 else 1
#     change_at = input_shape[time_dim] // 2
    
#     if len(input_shape) > 3:  # For 5D tensors (b,c,t,h,w)
#         x2[:, :, change_at:] = torch.randn_like(x2[:, :, change_at:])
#     else:  # For 3D tensors (b,t,c)
#         x2[:, change_at:] = torch.randn_like(x2[:, change_at:])
    
#     # Make condition tensors if needed
#     cond1 = cond2 = None
#     if is_cross and cond_shape:
#         cond1 = torch.randn(*cond_shape)
#         cond2 = cond1.clone()
#         cond2[:, change_at:] = torch.randn_like(cond2[:, change_at:])
    
#     # Process inputs
#     with torch.no_grad():
#         if is_cross:
#             print('cond2', cond1.shape)
#             print('x2', x1.shape)
#             out1 = module(x1, cond1)
#             out2 = module(x2, cond2)
#         else:
#             out1 = module(x1)
#             out2 = module(x2)
    
#     # Calculate differences
#     diffs = []
#     for t in range(input_shape[time_dim]):
#         if len(input_shape) > 3:
#             diff = torch.abs(out1[:, :, t] - out2[:, :, t]).mean().item()
#         else:
#             diff = torch.abs(out1[:, t] - out2[:, t]).mean().item()
#         diffs.append(diff)
    
#     # Check if causal
#     past_max_diff = max(diffs[:change_at])
#     print(f"Max difference in past timesteps: {past_max_diff:.6f}")
#     is_causal = past_max_diff < 1e-5
#     print(f"Is causal? {'Yes' if is_causal else 'No'}")
    
#     return is_causal, diffs

# def test_all_components():
#     """Test all components and the full model"""
#     # Setup dimensions
#     batch_size = 1
#     time_steps = 3
#     hidden_channels = 32
#     channels = 1
#     cond_channels = 1
#     height = width = 2
#     heads = 1
#     dim_head = 32
    
#     # Test full model
#     print("\nTesting full U-Net3D model...")
   
#     model = Unet3D(
#         input_size=height,
#         in_channels=channels,
#         z_channels=cond_channels,
#         init_hidden_channels=hidden_channels,
#         dim_mults=[1, 2],
#         num_res_blocks=1,
#         resnet_block_groups=1,        
#         attn_resolutions=[2],
#         attn_heads=heads,
#         attn_head_dim=dim_head,
#         use_linear_attn=False,
#         use_init_temporal_attn=True,
#         init_kernel_size=1,
#         dropout=0.0,
#         is_causal_selfattn=True,
#         is_causal_crossattn=True,
#         use_fourier_noise_embedding=False,
#         use_fourier_cond_embedding=False,
#     )   
    
#     disable_norm_layers(model)
    
#     # Test the model
#     x_shape = (batch_size, time_steps, channels, height, width)
#     noise_shape = (batch_size, time_steps)
#     cond_shape = (batch_size, time_steps, cond_channels, 32, 32)
    
#     torch.manual_seed(42)
#     x1 = torch.randn(*x_shape)
#     x2 = x1.clone()
#     noise = torch.ones(*noise_shape)
#     cond1 = torch.randn(*cond_shape)
#     cond2 = cond1.clone()
    
#     # Change future timesteps
#     change_at = 2
#     print('change_at', change_at)
#     #x2[:, change_at] = torch.randn_like(x2[:, change_at])
#     cond2[:, change_at] = torch.randn_like(cond2[:, change_at])
    
#     # Process
#     with torch.no_grad():
#         out1 = model(x1, noise, cond1)
#         out2 = model(x2, noise, cond2)

#     print(out1)
#     print(out2)
#     print(out1 - out2)
#     print(out1.shape)
    
#     # Calculate differences
#     diffs = []
#     for t in range(time_steps):
#         diff = torch.abs(out1[:, t] - out2[:, t]).mean().item()
#         diffs.append(diff)
#         print(f"Timestep {t}: Mean diff = {diff:.6f}")
    
#     past_max_diff = max(diffs[:change_at])
#     print(f"Max difference in past timesteps: {past_max_diff:.6f}")
#     is_causal = past_max_diff < 1e-5
#     print(f"Is full model causal? {'Yes' if is_causal else 'No'}")
    
    
#     # Plot results
#     # plt.figure(figsize=(12, 8))
#     # for i, (name, (_, diffs)) in enumerate(results.items()):
#     #     plt.subplot(3, 2, i+1)
#     #     plt.bar(range(len(diffs)), diffs)
#     #     plt.axvline(x=change_at-0.5, color='r', linestyle='--')
#     #     plt.title(name)
#     #     plt.tight_layout()
    
#     plt.savefig('causal_tests.png')
#     print("Results saved to causal_tests.png")
    
#     # Print summary
#     # print("\nSummary:")
#     # for name, (is_causal, _) in results.items():
#     #     print(f"{name}: {'Causal' if is_causal else 'Not Causal'}")
    
#     # # Identify the problem
#     # if not results["full_model"][0]:
#     #     if not results["temporal_causal"][0]:
#     #         print("Problem: Temporal self-attention causal mask is not working")
#     #     if not results["cross_causal"][0]:
#     #         print("Problem: Cross-attention causal mask is not working")
#     #     if not results["temporal_causal"][0] and not results["cross_causal"][0]:
#     #         print("Problem: Both attention mechanisms are not properly causal")
#     #     print("Check the mask implementation in attention blocks")
    
#     # return results

# if __name__ == "__main__":
#     test_all_components()