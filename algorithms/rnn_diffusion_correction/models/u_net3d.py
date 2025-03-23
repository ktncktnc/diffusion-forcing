from functools import partial
from typing import Optional
import torch
from torch import nn
from einops import rearrange
from omegaconf import DictConfig
from rotary_embedding_torch import RotaryEmbedding
from .embeddings import Timesteps, TimestepEmbedding, ConditionEmbedding
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
        out_channels: int = None
    ):
        super().__init__()

        # self.cond_channel = cfg.z_shape[0]
        self.cond_channel = z_channels
        self.is_causal_selfattn = cfg.is_causal_selfattn
        self.is_causal_crossattn = cfg.is_causal_crossattn

        self.use_init_temporal_attn = cfg.use_init_temporal_attn
        self.use_linear_attn = cfg.use_linear_attn
        self.attn_resolutions = cfg.attn_resolutions

        self.channel_mult = cfg.channel_mult
        self.init_hidden_channels = cfg.init_hidden_channels
        self.init_kernel_size = cfg.init_kernel_size

        self.num_heads = cfg.num_heads
        self.attn_head_dim = cfg.attn_head_dim

        self.num_res_blocks = cfg.num_res_blocks
        self.resnet_block_groups = cfg.resnet_block_groups

        resolution = input_size
        self.out_channel = out_channels if out_channels is not None else in_channels
        self.attn_resolutions = [resolution // res for res in list(self.attn_resolutions)]

        channels = [self.init_hidden_channels, *map(lambda m: self.init_hidden_channels * m, self.channel_mult)]
        in_out = list(zip(channels[:-1], channels[1:]))
        mid_channel = channels[-1]

        # emb_dim = self.noise_level_emb_dim

        init_padding = self.init_kernel_size // 2       
        self.init_conv = nn.Conv3d(
            in_channels,
            self.init_hidden_channels,
            kernel_size=(1, self.init_kernel_size, self.init_kernel_size),
            padding=(0, init_padding, init_padding),
        )

        self.noise_level_position_embedding = Timesteps(self.noise_level_dim)
        # self.noise_level_embedding = TimestepEmbedding(
        #     in_channels=self.noise_level_dim,
        #     time_embed_dim=self.condition_dim
        # )
        self.noise_level_embedding = nn.Sequential(
            nn.Linear(self.noise_level_dim, self.condition_dim),
            nn.SiLU(),
            nn.Linear(self.condition_dim, self.condition_dim),
        )

        self.cond_embedding = ConditionEmbedding(self.cond_channel, self.condition_dim)

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
        block_klass_resnet_noise = partial(
            ResnetBlock, groups=self.resnet_block_groups, emb_dim=self.condition_dim
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
        
        condition_crossattn_klass = partial(
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
                            # n Resnet blocks
                            block_klass_resnet_noise(dim_in, dim_out),
                            *(
                                block_klass_resnet_noise(dim_out, dim_out)
                                for _ in range(self.num_res_blocks - 1)
                            ),
                            # spatial attn
                            (
                                spatial_attn_klass(dim_out, use_linear=self.use_linear_attn and not is_last)
                                if use_attn
                                else nn.Identity()
                            ),
                            # temporal attn
                            temporal_attn_klass(dim_out) if use_attn else nn.Identity(),
                            # cross attn for hidden states conditions
                            condition_crossattn_klass(
                                query_dim=dim_out,
                                key_dim=self.condition_dim,
                                value_dim=self.condition_dim
                            ),
                        ),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

            curr_resolution *= 2 if not is_last else 1

        self.mid_block = UnetSequentialCondition(
            block_klass_resnet_noise(mid_channel, mid_channel),
            spatial_attn_klass(mid_channel),
            temporal_attn_klass(mid_channel),
            condition_crossattn_klass(
                query_dim=dim_out,
                key_dim=self.condition_dim,
                value_dim=self.condition_dim
            ),
            block_klass_resnet_noise(mid_channel, mid_channel),
        )

        for idx, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = idx == len(in_out) - 1
            use_attn = curr_resolution in self.attn_resolutions

            self.up_blocks.append(
                UnetSequentialCondition(
                    block_klass_resnet_noise(dim_out*2, dim_in),
                    *(
                        block_klass_resnet_noise(dim_in, dim_in)
                        for _ in range(self.num_res_blocks - 1)
                    ),
                    (
                        spatial_attn_klass(dim_in, use_linear=self.use_linear_attn and idx > 0) 
                        if use_attn 
                        else nn.Identity()
                    ),
                    temporal_attn_klass(dim_in) if use_attn else nn.Identity(),
                    condition_crossattn_klass(
                        query_dim=dim_in,
                        key_dim=self.condition_dim,
                        value_dim=self.condition_dim
                    ),
                    Upsample(dim_in) if not is_last else nn.Identity(),
                )
            )
            curr_resolution //= 2 if not is_last else 1

        self.out = nn.Sequential(
            block_klass(self.init_hidden_channels*2, self.init_hidden_channels), 
            nn.Conv3d(self.init_hidden_channels, self.out_channel, 1)
        )

    @property
    def condition_dim(self):
        return self.init_hidden_channels * 4
    
    @property
    def noise_level_dim(self):
        return max(self.condition_dim // 4, 32)

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor],
        t: torch.Tensor
    ):
        x = rearrange(x, "b t c h w -> b c t h w").contiguous()

        # TODO: patchify x
        t_embed = self.noise_level_position_embedding(t) # (b, t, emb_dim)
        difffusion_time_emb = self.noise_level_embedding(t_embed) # (b, t, emb_dim)
        cond_emb = None

        if self.cond_embedding is not None:
            if cond is None:
                raise ValueError("External condition is required, but not provided.")
            cond_emb = self.cond_embedding(cond) # (b, t, emb_dim)

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



import torch
from algorithms.rnn_diffusion_correction.models.u_net3d import Unet3D
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import numpy as np


def disable_normalization_layers(model):
    """Disable all normalization layers in the model"""
    for module in model.modules():
        # Check for normalization layers and set them to identity
        if hasattr(module, 'weight') and hasattr(module, 'running_mean'):
            # For BatchNorm layers
            if hasattr(module, 'track_running_stats'):
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
            
            # For other normalization layers, we could add more conditions if needed

def disable_norm_layers(model):
    """Replace all normalization layers with identity"""
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
                              torch.nn.LayerNorm, torch.nn.GroupNorm, torch.nn.InstanceNorm1d,
                              torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d)):
            # Save the original state and replace with identity
            module._original_forward = module.forward
            module.forward = lambda x: x
    return model

def create_config():
    """Create a simple configuration for testing"""
    cfg = {
        "is_causal_selfattn": True,
        "is_causal_crossattn": True,
        "use_init_temporal_attn": True,
        "use_linear_attn": False,
        "attn_resolutions": [8, 16, 32, 64],
        "channel_mult": [1, 2, 4, 8],
        "init_hidden_channels": 48,
        "init_kernel_size": 3,
        "num_heads": 4,
        "attn_head_dim": 32,
        "num_res_blocks": 2,
        "resnet_block_groups": 8
    }
    return OmegaConf.create(cfg)

def visualize_attention_patterns(model, x, cond, t):
    """Run forward pass and extract attention patterns from key modules"""
    # Store original forward methods
    original_forwards = {}
    attention_maps = {}
    
    # Monkey patch to capture attention patterns
    def hook_attention(module, name):
        original_forward = module.forward
        
        def patched_forward(*args, **kwargs):
            result = original_forward(*args, **kwargs)
            if hasattr(module, 'attn'):
                # Get the last attention map
                attention_maps[name] = module.attn.last_attn_map.detach().cpu()
            return result
        
        original_forwards[name] = original_forward
        module.forward = patched_forward
    
    # Apply hooks to relevant modules
    for name, module in model.named_modules():
        if 'UnetTemporalAttentionBlock' in module.__class__.__name__:
            hook_attention(module, name)
    
    # Run forward pass
    with torch.no_grad():
        output = model(x, cond, t)
    
    # Restore original forward methods
    for name, forward in original_forwards.items():
        modules_path = name.split('.')
        current_module = model
        for path in modules_path[:-1]:
            current_module = getattr(current_module, path)
        setattr(current_module, modules_path[-1], forward)
    
    # Plot attention maps
    fig, axes = plt.subplots(len(attention_maps), 1, figsize=(10, 4*len(attention_maps)))
    if len(attention_maps) == 1:
        axes = [axes]
        
    for i, (name, attn_map) in enumerate(attention_maps.items()):
        # Get the first batch, first head
        attn_map = attn_map[0, 0].numpy()
        im = axes[i].imshow(attn_map, cmap='viridis')
        axes[i].set_title(f"Attention Pattern: {name}")
        axes[i].set_xlabel("Key position (time)")
        axes[i].set_ylabel("Query position (time)")
        fig.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.savefig("attention_patterns.png")
    print(f"Saved attention visualization to attention_patterns.png")
    
    return attention_maps


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Create test inputs
    batch_size = 2
    time_steps = 8
    channels = 3
    height = 64
    width = 64
    
    # Creating random input tensor
    x = torch.randn(batch_size, time_steps, channels, height, width).to(device)
    
    # Create condition tensor
    z_channels = 4
    cond = torch.randn(batch_size, time_steps, z_channels, height, width).to(device)
    
    # Create diffusion timesteps
    t = torch.randint(0, 1000, (batch_size, time_steps)).to(device)
    
    # Create model
    cfg = create_config()
    input_size = height
    model = Unet3D(
        cfg=cfg,
        input_size=input_size,
        in_channels=channels,
        z_channels=z_channels
    ).to(device)
    print('model', model)
    
    # Disable normalization layers
    disable_norm_layers(model)
    
    # Print model information
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test causal attention
    print("Testing forward pass with causal attention...")
    with torch.no_grad():
        output = model(x, cond, t)
        # print('output', output)

        x[:, 1, ...] = 100
        output = model(x, cond, t)
        # print('output', output)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Forward pass successful!")
    
    # Visualize attention patterns to verify causality
    # print("Visualizing attention patterns...")
    # attention_maps = visualize_attention_patterns(model, x, cond, t)
    
    # # Check if attention maps are causal (lower triangular)
    # for name, attn_map in attention_maps.items():
    #     attn_map = attn_map[0, 0].numpy()  # First batch, first head
    #     is_causal = np.allclose(np.triu(attn_map, k=1), 0, atol=1e-5)
    #     print(f"Attention map '{name}' is {'causal' if is_causal else 'NOT causal'}")

if __name__ == "__main__":
    main()