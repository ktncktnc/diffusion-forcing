"""
Originally inspired by impl at https://github.com/facebookresearch/DiT/blob/main/models.py

Modified by Haoyu Lu, for video diffusion transformer
"""
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# DiT: https://github.com/facebookresearch/DiT/blob/main/models.py
# 
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
import collections
from torch.nn.attention import SDPBackend, sdpa_kernel
from typing import Type
from functools import partial
from timm.layers.helpers import to_2tuple
from einops import rearrange, reduce
import torch.nn.functional as F


def modulate(x, shift, scale, T):
    N, M = x.shape[-2], x.shape[-1]
    B = scale.shape[0]
    x = rearrange(x, '(b t) n m-> b (t n) m',b=B,t=T,n=N,m=M)
    x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    x = rearrange(x, 'b (t n) m-> (b t) n m',b=B,t=T,n=N,m=M)
    return x


def generate_mask_for_cross_attention(n_frames, tokens_per_frame, device):
    """
    Generate causal mask for cross-attention block.
    For each query position, it can only attend to conditioning tokens
    corresponding to its frame or earlier frames.
    
    Args:
        n_frames: Number of frames in the sequence
        tokens_per_frame: Number of tokens per frame
        
    Returns:
        Binary mask tensor with shape (n_frames*tokens_per_frame, n_frames)
    """
    N_q = n_frames * tokens_per_frame
    N_k = n_frames
    
    # q_indices = torch.arange(N_q).unsqueeze(1)
    # k_indices = torch.arange(N_k).unsqueeze(0)
    # mask = (q_indices // n_frames) >= (k_indices // tokens_per_frame)
    q_frame_indices = torch.arange(N_q).div(tokens_per_frame, rounding_mode='floor').unsqueeze(1)
    k_indices = torch.arange(N_k).unsqueeze(0)
    
    # Create mask: 1 if query can attend to key, 0 otherwise
    mask = (q_frame_indices >= k_indices).float()
    return mask.bool().to(device)

def generate_mask_for_self_attention(n_frames, tokens_per_frame, device):
    """
    Generate causal mask for self-attention block.
    Each token can attend to all tokens in its frame and all tokens in previous frames.
    
    Args:
        n_frames: Number of frames in the sequence
        tokens_per_frame: Number of tokens per frame
        
    Returns:
        Binary mask tensor with shape (n_frames*tokens_per_frame, n_frames*tokens_per_frame)
    """
    N = n_frames * tokens_per_frame
    q_indices = torch.arange(N).unsqueeze(1)
    k_indices = torch.arange(N).unsqueeze(0)
    mask = (q_indices // tokens_per_frame) >= (k_indices // tokens_per_frame)
    return mask.bool().to(device)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        assert W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
    

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks

    NOTE: When use_conv=True, expects 2D NCHW tensors, otherwise N*C expected.
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])



    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self._reshape_for_multihead = lambda x: rearrange(x, 'b n (h d) -> b h n d', h=self.num_heads)
        self.cpu_backends = [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.MATH,
            SDPBackend.EFFICIENT_ATTENTION,
        ]

        device_properties = torch.cuda.get_device_properties(torch.device("cuda"))

        if device_properties.major >= 8 and device_properties.minor == 0:
            # A100 GPU
            self.cuda_backends = [SDPBackend.FLASH_ATTENTION]
        else:
            # Non-A100 GPU
            self.cuda_backends = [
                SDPBackend.MATH,
                SDPBackend.EFFICIENT_ATTENTION,
            ]

    def forward(
        self, 
        q: torch.Tensor,
        k: torch.Tensor, 
        v: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        B, N_q, D = q.shape
        N_k = k.shape[1]  

        q = self._reshape_for_multihead(self.q_proj(q)) # (B, H, N_q, D)
        k = self._reshape_for_multihead(self.k_proj(k)) # (B, H, N_k, D)
        v = self._reshape_for_multihead(self.v_proj(v)) # (B, H, N_k, D)

        q= self.q_norm(q) # (B, H, N_q, D)
        k = self.k_norm(k) # (B, H, N_k, D)

        backends = (
            (
                [SDPBackend.MATH]
                if q.shape[0] >= 65536
                else (
                    self.cuda_backends
                    if mask is None
                    else [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]
                )
            )
            if q.is_cuda
            else self.cpu_backends
        )
        q, k, v = map(lambda t: t.contiguous(), (q, k, v))
        with sdpa_kernel(backends=backends):
            x = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                dropout_p=self.attn_drop,
                attn_mask=mask,
                scale=self.scale,
            )

        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

class SelfAttention(Attention):
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        return super().forward(x, x, x, mask=mask)


class CrossAttention(Attention):
    def forward(        
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        return super().forward(
            q=q,
            k=k,
            v=v,
            mask=mask
        )
    

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        # if t.dim() == 1:
        #     t = t.unsqueeze(1)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
        


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class ZtoVDTConvAdapter(nn.Module):
    def __init__(self, input_channels, hidden_size):
        super().__init__()
        
        # Use convolutional layers to process spatial information
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        )
        
        # Final projection to get to z_size
        self.projection = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, z_rnn):
        B, T, C, H, W = z_rnn.shape
        z_rnn = rearrange(z_rnn, 'b t c h w -> (b t) c h w')
        z_conv = self.conv(z_rnn)  # Shape: (B*T, hidden_size, 1, 1)
        z_conv = z_conv.flatten(1)  # Shape: (B*T, hidden_size)
        z_proj = self.projection(z_conv)  # Shape: (B*T, hidden_size)
        z_vdt = rearrange(z_proj, '(b t) d -> b t d', b=B, t=T)
        
        return z_vdt

#################################################################################
#                                 Core VDT Model                                #
#################################################################################

class CrossAttnVDTBlock(nn.Module):
    """
    A VDT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        is_causal_crossattn=True,
        is_causal_selfattn=True,
        **block_kwargs
    ):
        """
        Include 1) Cross Attention Block for hidden state condition
                2) Self Attention Block for temporal frame attention
        Note that spatial attention in one frame is already included in the self attention blocks.
        """
        super().__init__()
        # Cross Attention Block for hidden state condition
        self.is_causal_crossattn = is_causal_crossattn
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.condition_crossattn = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.fc1 = nn.Linear(hidden_size, hidden_size, bias=True)

        # Self Attention Block for temporal frame attention
        self.is_causal_selfattn = is_causal_selfattn
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.frame_selfattn = SelfAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=True)

        # MLP Block
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        # AdaLN-Zero modulation for diffusion time
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )      

    def forward(self, x, z, t, n_frames):
        """
        x: hidden state, ((B, T), N, D) where B is batch size, T is n_frames, N is sequence length of 1 frame
        z: conditioning input, typically encoder output
        t: diffusion timestep embedding
        n_frames: number of frames in the sequence
        """
        # Diffusion time, condition using AdaLN-Zero
        
        shift_self, scale_self, gate_self, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=-1)
        T = n_frames
        K, N, M = x.shape
        B = K // T
        # Cross Attention Block for hidden state condition
        cross_mask = generate_mask_for_cross_attention(n_frames, N, x.device) if self.is_causal_crossattn else None
        norm_x = self.norm1(x)
        norm_x = rearrange(norm_x, '(b t) n d -> b (t n) d',b=B, t=T, n=N)
        cross_attn = self.condition_crossattn(q=norm_x, k=z, v=z,mask=cross_mask) # TODO: check dimension of z
        cross_attn = rearrange(cross_attn, 'b (t n) d -> (b t) n d',b=B, t=T, n=N)
        x = x + self.fc1(cross_attn)

        # Self Attention Block for temporal frame attention
        self_mask = generate_mask_for_self_attention(n_frames, N, x.device) if self.is_causal_selfattn else None
        norm_x = modulate(self.norm2(x), shift_self, scale_self, T)
        norm_x = rearrange(norm_x, '(b t) n d -> b (t n) d',b=B, t=T, n=N)

        self_attn = self.frame_selfattn(x=norm_x, mask=self_mask)
        self_attn = gate_self.unsqueeze(1) * self.fc2(self_attn)
        self_attn = rearrange(self_attn, 'b (t n) m-> (b t) n m',b=B,t=T,n=N,m=M)

        x = x + self_attn

        # MLP Block
        norm_x = modulate(self.norm3(x), shift_mlp, scale_mlp, T)
        mlp = self.mlp(norm_x)
        mlp = rearrange(mlp, '(b t) n m-> b (t n) m',b=B,t=T,n=N,m=M)
        mlp = gate_mlp.unsqueeze(1) * mlp
        mlp = rearrange(mlp, 'b (t n) m-> (b t) n m',b=B,t=T,n=N,m=M)
        x = x + mlp

        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    

class FinalLayer(nn.Module):
    """
    The final layer of VDT.
    """
    def __init__(self, hidden_size, num_heads, patch_size, out_channels, is_causal_crossattn=True):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.is_causal_crossattn = is_causal_crossattn
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.cross_attn = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True)

    def forward(self, x, z, t, num_frames):
        K, N, M = x.shape
        T = num_frames
        B = K // T
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        norm_x = modulate(self.norm_final(x), shift, scale, num_frames)

        norm_x = rearrange(norm_x, '(b t) n d -> b (t n) d',b=B, t=T, n=N)
        mask = generate_mask_for_cross_attention(num_frames, N, x.device) if self.is_causal_crossattn else None
        cross_attn = self.cross_attn(q=norm_x, k=z, v=z, mask=mask)
        cross_attn = rearrange(cross_attn, 'b (t n) d -> (b t) n d',b=B, t=T, n=N)
        x = self.linear(x)
        return x


class VDT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        cfg,
        input_size=32,
        in_channels=4,
        z_channels=4,
        # patch_size=2,
        # hidden_size=1152,
        # depth=28,
        # num_heads=16,
        # mlp_ratio=4.0,
        # num_frames=16,
        # is_causal_crossattn=True,
        # is_causal_selfattn=True,
        # **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        self.input_size = input_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = cfg.hidden_size
        self.cond_channels = z_channels
        self.num_frames = cfg.get('num_frames', 32)
        self.depth = cfg.depth
        self.mlp_ratio = cfg.mlp_ratio

        self.patch_size = cfg.patch_size
        self.num_heads = cfg.num_heads

        self.is_causal_crossattn = cfg.is_causal_crossattn
        self.is_causal_selfattn = cfg.is_causal_selfattn

        # Tokenizers
        self.x_embedder = PatchEmbed(input_size, self.patch_size, in_channels, self.hidden_size, bias=True)
        self.diffusion_t_embedder = TimestepEmbedder(self.hidden_size)
        num_patches = self.x_embedder.num_patches

        # Z embedding
        self.cond_channels = self.cond_channels
        self.cond_embedding = ZtoVDTConvAdapter(
            input_channels=self.cond_channels,
            hidden_size=self.hidden_size
        )

        # Spatial positional embeddings
        # Will use fixed sin-cos embedding:
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.hidden_size), requires_grad=False)
        
        # Temporal positional embeddings
        self.num_frames = self.num_frames
        self.temporal_frame_embed = nn.Parameter(torch.zeros(1, self.num_frames, self.hidden_size), requires_grad=False)
        self.temporal_frame_drop = nn.Dropout(p=0)

        self.blocks = nn.ModuleList([
            CrossAttnVDTBlock(
                hidden_size=self.hidden_size, 
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio, 
                is_causal_crossattn=self.is_causal_crossattn, 
                is_causal_selfattn=self.is_causal_selfattn
            ) 
            for _ in range(self.depth)
        ])
        self.final_layer = FinalLayer(
            self.hidden_size, 
            self.num_heads, 
            self.patch_size, 
            self.out_channels,
            is_causal_crossattn=self.is_causal_crossattn
        )
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        spatial_pos_embed = get_2d_sincos_pos_embed(self.spatial_pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.spatial_pos_embed.data.copy_(torch.from_numpy(spatial_pos_embed).float().unsqueeze(0))

        grid_num_frames = np.arange(self.num_frames, dtype=np.float32)
        temporal_frame_embed = get_1d_sincos_pos_embed_from_grid(self.spatial_pos_embed.shape[-1], grid_num_frames)
        self.temporal_frame_embed.data.copy_(torch.from_numpy(temporal_frame_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.diffusion_t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.diffusion_t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in VDT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        Unpatchify an image tensor.
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('n h w p q c-> n c h p w q', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, cond, t):
        """
        Forward pass of VDT.
        x: (B, T, C, H, W) tensor of spatial inputs (images or latent representations of images)
        z: (B, T, D) tensor of conditioning inputs
        t: (B,) tensor of diffusion timesteps
        """
        
        B, T, C, W, H = x.shape # 32 16 4 8 8 
        x = x.contiguous().view(-1, C, W, H)

        # Patchify input into tokens and add spatial positional embeddings
        x = self.x_embedder(x) + self.spatial_pos_embed  # (N, K, D), where N = B*T, K: number of tokens, D: token dimension

        # Temporal embed: reshape to make spatial dimension as batch 
        x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T)
        # Resizing time embeddings in case they don't match
        x = x + self.temporal_frame_embed[:, :T]
        # Dropout
        x = self.temporal_frame_drop(x)
        # Spatial embed: Reshape back
        x = rearrange(x, '(b n) t m -> (b t) n m',b=B,t=T)
        
        # Diffusion timestep embedding
        t = self.diffusion_t_embedder(t) # (B, D)

        # Z embedding
        cond = self.cond_embedding(cond) # (B, T, D)

        for block in self.blocks:
            x = block(x, cond, t, T) # ((B,T), N, D)

        x = self.final_layer(x, cond, t, T) # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x) # (N, out_channels, H, W)
        x = x.view(B, T, x.shape[-3], x.shape[-2], x.shape[-1])
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of VDT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   VDT Configs                                  #
#################################################################################

def VDT_L_2(**kwargs):
    return VDT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def VDT_S_2(**kwargs):
    return VDT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def VDT_S_1(**kwargs):
    return VDT(depth=1, hidden_size=144, patch_size=2, num_heads=1, **kwargs)


VDT_models = {
    'VDT-L/2':  VDT_L_2,
    'VDT-S/2':  VDT_S_2, 
    'VDT-S/1':  VDT_S_1,  
}



def main():
    """
    A simple test function to verify the VDT model with causal temporal attention.
    Tests input/output shapes and runs a basic forward pass.
    """
    import torch
    import time
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Test parameters
    batch_size = 1
    num_frames = 7
    input_size = 32
    in_channels = 1
    z_channel = 16
    model_type = 'VDT-S/1'  # Use smaller model for testing
    
    # Choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    print(f"Creating {model_type} model...")
    model = VDT(
        depth=12,
        patch_size=2,
        z_channels=z_channel,
        input_size=input_size,
        in_channels=in_channels,
        num_frames=40,
        hidden_size=144,
        num_heads=4
    )
    model = model.to(device)
    
    # Print model size
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {num_params / 1e6:.2f}M parameters")
    
    # Create sample input
    print(f"Creating sample input with shape: [{batch_size}, {num_frames}, {in_channels}, {input_size}, {input_size}]")
    x = torch.randn(batch_size, num_frames, in_channels, input_size, input_size, device=device)
    z = torch.randn(batch_size, num_frames, z_channel, 32, 32, device=device)
    t = torch.randint(10,100, (batch_size,), device=device).long()  # Timestep tokens
    
    # Run forward pass with timing
    model.eval()
    with torch.no_grad():
        print("Running forward pass...")
        start_time = time.time()
        output = model(x, z, t)
        end_time = time.time()
    
    # Print shape information
    print("\n--- Shape Information ---")
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Forward pass time: {end_time - start_time:.4f} seconds")
    print('\nOutput:', output)
    # Verification
    expected_shape = (batch_size, num_frames, in_channels, input_size, input_size)
    if output.shape == expected_shape:
        print("\n✅ SUCCESS: Output shape matches expected shape!")
    else:
        print(f"\n❌ ERROR: Output shape {output.shape} doesn't match expected shape {expected_shape}!")
    
    print("\nTest completed.")


if __name__ == "__main__":
    main()