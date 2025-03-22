from typing import Type

from torch import nn as nn
from .sin_emb import SinusoidalPosEmb   


class ResBlock2d(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, activation: Type[nn.Module] = nn.ReLU):
        super(ResBlock2d, self).__init__()
        self.activation = activation()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = self.activation(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class ResBlock1d(nn.Module):
    def __init__(self, in_planes, planes, activation: Type[nn.Module] = nn.ReLU):
        super().__init__()
        self.activation = activation()
        self.mlp = nn.Sequential(
            nn.Linear(in_planes, planes),
            self.activation,
            nn.Linear(planes, planes),
        )

        self.shortcut = nn.Identity() if in_planes == planes else nn.Linear(in_planes, planes)

    def forward(self, x):
        out = self.activation(self.shortcut(x) + self.mlp(x))
        return out


class ResBlockWrapper(nn.Module):
    def __init__(self, model: nn.Module, in_planes=None, out_planes=None):
        super().__init__()
        self.model = model
        self.shortcut = (
            nn.Identity()
            if in_planes == out_planes
            else nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return self.model(x) + self.shortcut(x)



from typing import Type, Union, Tuple

import torch
from torch import nn as nn

class ConditionalResBlock2d(nn.Module):
    expansion = 1

    def __init__(self, 
                 in_planes: int, 
                 planes: int, 
                 cond_channels: int,
                 stride: int = 1, 
                 activation: Type[nn.Module] = nn.ReLU,
                 use_film: bool = True):
        """
        Conditional ResBlock with FiLM conditioning or concat conditioning.
        
        Args:
            in_planes: Number of input channels
            planes: Number of output channels
            cond_channels: Number of condition channels
            stride: Stride for convolution
            activation: Activation function to use
            use_film: Whether to use FiLM conditioning (True) or concatenation (False)
        """
        super(ConditionalResBlock2d, self).__init__()
        self.activation = activation()
        self.use_film = use_film
        
        if use_film:
            # FiLM conditioning
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            
            # FiLM conditioning layers (scale and shift)
            self.film1 = nn.Conv2d(cond_channels, planes*2, kernel_size=1, stride=1)
            self.film2 = nn.Conv2d(cond_channels, planes*2, kernel_size=1, stride=1)
        else:
            # Concatenation conditioning
            self.conv1 = nn.Conv2d(in_planes + cond_channels, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.conv2 = nn.Conv2d(planes + cond_channels, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
            )

    def _apply_film(self, x, cond, film_layer):
        """Apply FiLM conditioning (feature-wise linear modulation)"""
        film_params = film_layer(cond)
        gamma, beta = torch.chunk(film_params, 2, dim=1)
        return gamma * x + beta
        
    def forward(self, x, cond):
        """
        Forward pass with conditioning
        
        Args:
            x: Input tensor of shape (batch, in_planes, height, width)
            cond: Conditioning tensor of shape (batch, cond_channels, height, width)
        """
        if self.use_film:
            # FiLM conditioning
            out = self.conv1(x)
            out = self._apply_film(out, cond, self.film1)
            out = self.activation(out)
            
            out = self.conv2(out)
            out = self._apply_film(out, cond, self.film2)
            
            out += self.shortcut(x)
            out = self.activation(out)
            
        else:
            # Concatenation conditioning
            out = torch.cat([x, cond], dim=1)
            out = self.activation(self.conv1(out))
            
            out = torch.cat([out, cond], dim=1)
            out = self.conv2(out)
            
            out += self.shortcut(x)
            out = self.activation(out)
            
        return out


class ConditionalResNet2d(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 cond_channels: int,
                 time_embedding_dim: int = 16,
                 num_blocks: int = 3,
                 activation: Type[nn.Module] = nn.ReLU,
                 use_film: bool = True):
        """
        Conditional ResNet with multiple conditional ResBlocks incorporating diffusion timestep
        
        Args:
            in_channels: Number of input channels
            hidden_channels: Number of hidden channels in ResBlocks
            out_channels: Number of output channels
            cond_channels: Number of condition channels
            time_embedding_dim: Dimension of timestep embeddings
            num_blocks: Number of ResBlocks to use
            activation: Activation function to use
            use_film: Whether to use FiLM conditioning or concatenation
        """
        super(ConditionalResNet2d, self).__init__()
        
        # Replace time embedding with Sinusoidal embedding
        self.time_embed = SinusoidalPosEmb(time_embedding_dim)
        
        # Project time embedding to spatial dimensions for combination with spatial condition
        self.time_to_cond = nn.Sequential(
            nn.Linear(time_embedding_dim, cond_channels),
            activation()
        )
        
        self.input_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(
                ConditionalResBlock2d(
                    hidden_channels, 
                    hidden_channels, 
                    cond_channels,
                    activation=activation,
                    use_film=use_film
                )
            )
            
        self.output_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.activation = activation()
        
    def forward(self, x, t, cond):
        """
        Forward pass with conditioning and diffusion timestep
        
        Args:
            x: Input tensor of shape (batch, in_channels, height, width)
            cond: Conditioning tensor of shape (batch, cond_channels, height, width)
            t: Diffusion timestep/noise level of shape (batch,), integer ranging from 0 to 1000
        """
        batch, _, h, w = cond.shape
        
        # Process timestep using sinusoidal embedding
        t_embed = self.time_embed(t)  # [batch, time_embedding_dim]
        
        # Map time embedding to condition space and expand spatially
        t_cond = self.time_to_cond(t_embed)  # [batch, cond_channels]
        t_cond = t_cond.view(batch, -1, 1, 1).expand(-1, -1, h, w)
        # Combine spatial condition with timestep condition
        combined_cond = cond + t_cond
        
        # Process through network
        out = self.input_conv(x)
        out = self.activation(out)
        
        for block in self.blocks:
            out = block(out, combined_cond)
            
        out = self.output_conv(out)
        
        return out
    

class DenoisingNetworkWrapper(nn.Module):
    def __init__(self, x_channel, hidden_channel, z_channel, num_blocks=2):
        super().__init__()
        self.conditional_resnet = ConditionalResNet2d(
            in_channels=x_channel,
            hidden_channels=hidden_channel,
            out_channels=x_channel,
            cond_channels=z_channel,
            num_blocks=num_blocks,
        )
        self.output_conv = nn.Conv2d(x_channel, x_channel, 1, padding=0)
    
    def forward(self, x, t, z_cond):
        """
        Forward pass that handles multiple inputs
        
        Args:
            x: Noised input tensor
            t: Timestep tensor
            z_cond: Conditioning tensor
        """
        x = self.conditional_resnet(x, t, z_cond)
        x = self.output_conv(x)
        return x
    

def main():
    """
    Simple test function for the ConditionalResNet2d model.
    Creates a small model, generates random input and condition tensors,
    and prints the output shape.
    """
    import torch
    
    # Model parameters
    in_channels = 3
    hidden_channels = 64
    out_channels = 3
    cond_channels = 4
    num_blocks = 2
    use_film = True
    
    # Create model
    model = ConditionalResNet2d(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        cond_channels=cond_channels,
        num_blocks=num_blocks,
        use_film=use_film
    )
    
    # Generate random input and condition
    batch_size = 4
    height, width = 32, 32
    x = torch.randn(batch_size, in_channels, height, width)
    cond = torch.randn(batch_size, cond_channels, height, width)
    t = torch.randint(0, 100, (batch_size,))
    
    # Forward pass
    print(f"Input shape: {x.shape}")
    print(f"Condition shape: {cond.shape}")
    
    # Run model
    with torch.no_grad():
        output = model(x, cond, t)
    
    print(f"Output shape: {output.shape}")
    print(f"Model parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test with different conditioning mode
    model_concat = ConditionalResNet2d(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        cond_channels=cond_channels,
        num_blocks=num_blocks,
        use_film=False  # Use concatenation instead of FiLM
    )
    
    with torch.no_grad():
        output_concat = model_concat(x, cond, t)
    
    print(f"Output shape (concatenation mode): {output_concat.shape}")
    
    # Verify FiLM and concatenation conditioning work with similar output shapes
    assert output.shape == output_concat.shape, "Output shapes should match between conditioning modes"
    print("Test passed: Both conditioning modes produce same shape output")


if __name__ == "__main__":
    main()