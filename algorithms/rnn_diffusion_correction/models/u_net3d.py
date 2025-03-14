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
        x_shape: torch.Size,
        max_tokens: int,
        cond_dim: int,
        use_causal_selfattn_mask=True,
        use_causal_crossattn_mask=True,
    ):
        super().__init__()

        self.cfg = cfg
        self.x_shape = x_shape
        self.cond_dim = cond_dim
        self.use_causal_selfattn_mask = use_causal_selfattn_mask
        self.use_causal_crossattn_mask = use_causal_crossattn_mask

        dim = cfg.network_size
        init_dim = dim
        channels, resolution, *_ = x_shape
        out_dim = channels

        num_res_blocks = cfg.num_res_blocks
        resnet_block_groups = cfg.resnet_block_groups
        dim_mults = cfg.dim_mults
        attn_resolutions = [resolution // res for res in list(cfg.attn_resolutions)]
        attn_dim_head = cfg.attn_dim_head

        attn_heads = cfg.attn_heads
        use_linear_attn = cfg.use_linear_attn
        use_init_temporal_attn = cfg.use_init_temporal_attn
        init_kernel_size = cfg.init_kernel_size
        dropout = cfg.dropout

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        mid_dim = dims[-1]

        emb_dim = self.noise_level_emb_dim

        init_padding = init_kernel_size // 2       
        self.init_conv = nn.Conv3d(
            channels,
            init_dim,
            kernel_size=(1, init_kernel_size, init_kernel_size),
            padding=(0, init_padding, init_padding),
        )

        self.noise_level_pos_embedding = StochasticTimeEmbedding(
            dim=self.noise_level_dim,
            time_embed_dim=self.noise_level_emb_dim,
            use_fourier=self.cfg.get("use_fourier_noise_embedding", False),
        )

        self.cond_embedding = ZtoVDTConvAdapter(
            self.cond_dim,
            self.noise_level_emb_dim,
            use_fourier=self.cfg.get("use_fourier_cond_embedding", False),
        )

        self.rotary_time_pos_embedding = RotaryEmbedding(dim=attn_dim_head)

        self.init_temporal_attn = (
            UnetTemporalAttentionBlock(
                dim=init_dim,
                heads=attn_heads,
                dim_head=attn_dim_head,
                is_causal=False,
                rotary_emb=self.rotary_time_pos_embedding,
            )
            if use_init_temporal_attn
            else IdentityWithExtraArgs()
        )

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)
        block_klass_noise = partial(
            ResnetBlock, groups=resnet_block_groups, emb_dim=emb_dim
        )
        spatial_attn_klass = partial(
            UnetSpatialAttentionBlock, heads=attn_heads, dim_head=attn_dim_head
        )

        temporal_attn_klass = partial(
            UnetTemporalAttentionBlock,
            heads=attn_heads,
            dim_head=attn_dim_head,
            is_causal=self.use_causal_crossattn_mask, #TODO:
            rotary_emb=self.rotary_time_pos_embedding,
        )
        
        crossattn_klass = partial(
            UnetTemporalCrossAttentionBlock,
            heads=attn_heads,
            dim_head=attn_dim_head,
            is_causal=self.use_causal_crossattn_mask, #TODO:
            rotary_emb=self.rotary_time_pos_embedding,
        )

        curr_resolution = 1
        for idx, (dim_in, dim_out) in enumerate(in_out):
            is_last = idx == len(in_out) - 1
            use_attn = curr_resolution in attn_resolutions

            self.down_blocks.append(
                nn.ModuleList(
                    [
                        UnetSequentialCondition(
                            block_klass_noise(dim_in, dim_out),
                            *(
                                block_klass_noise(dim_out, dim_out)
                                for _ in range(num_res_blocks - 1)
                            ),
                            (
                                spatial_attn_klass(
                                    dim_out,
                                    use_linear=use_linear_attn and not is_last,
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
            use_attn = curr_resolution in attn_resolutions

            self.up_blocks.append(
                UnetSequentialCondition(
                    block_klass_noise(dim_out * 2, dim_in),
                    *(
                        block_klass_noise(dim_in, dim_in)
                        for _ in range(num_res_blocks - 1)
                    ),
                    (
                        spatial_attn_klass(
                            dim_in, use_linear=use_linear_attn and idx > 0
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

        self.out = nn.Sequential(block_klass(dim * 2, dim), nn.Conv3d(dim, out_dim, 1))

    @property
    def noise_level_emb_dim(self):
        return self.cfg.network_size * 4

    @property
    def external_cond_emb_dim(self):
        return self.cfg.network_size * 2 if self.external_cond_dim else 0

    def forward(
        self,
        x: torch.Tensor,
        noise_levels: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        external_cond_mask: Optional[torch.Tensor] = None,
    ):
        x = rearrange(x, "b t c h w -> b c t h w").contiguous()

        # TODO: patchify x

        difffusion_time_emb = self.noise_level_pos_embedding(noise_levels) # (b, t, emb_dim)
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
    
    
    @property
    def noise_level_emb_dim(self):
        return self.cfg.network_size * 4
    
    @property
    def noise_level_dim(self):
        return max(self.noise_level_emb_dim // 4, 32)


if __name__ == "__main__":
    from omegaconf import OmegaConf
    
    # Create a simple configuration
    cfg = OmegaConf.create({
        'network_size': 32,
        'num_res_blocks': 2,
        'resnet_block_groups': 8,
        'dim_mults': [1, 2, 4, 8],
        'attn_resolutions': [32, 64, 128, 256],
        'attn_dim_head': 32,
        'attn_heads': 4,
        'use_linear_attn': False,
        'use_init_temporal_attn': True,
        'init_kernel_size': 7,
        'dropout': 0.0,
        'use_fourier_noise_embedding': True,
        'use_fourier_cond_embedding': False,
    })
    
    # Create test tensors
    batch_size = 2
    seq_length = 16
    channels = 3
    height = 32
    width = 32
    cond_dim = 16
    
    x = torch.randn(batch_size, seq_length, channels, height, width)
    noise_levels = torch.randn(batch_size, seq_length)
    cond = torch.randn(batch_size, seq_length, 3, cond_dim, cond_dim)  # Assuming 16-dim condition
    
    # Initialize and test model
    x_shape = (channels, seq_length, height, width)
    max_tokens = seq_length
    
    print("Initializing model...")
    model = Unet3D(
        cfg=cfg,
        x_shape=x_shape,
        max_tokens=max_tokens,
        cond_dim=3
    )
    
    print("Running forward pass...")
    output = model(x, noise_levels, cond)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Model test complete!")
