import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

class SwiGLU(nn.Module):
    """
    SwiGLU activation function as described in the paper:
    "GLU Variants Improve Transformer"
    """
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
        
    def forward(self, x):
        # Split the tensor along the specified dimension
        # For Conv2d outputs, dim=1 (channel dimension)
        # For Linear outputs, dim=-1 (last dimension)
        dim_size = x.size(self.dim)
        a, b = torch.split(x, dim_size // 2, dim=self.dim)
        return a * torch.sigmoid(b)

class SwiGLUConv2d(nn.Module):
    """
    Convolutional layer with SwiGLU activation
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        # Double the output channels since SwiGLU will halve them
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels * 2, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding
        )
        # Use SwiGLU module with dim=1 for Conv2d outputs (channel dimension)
        self.activation = SwiGLU(dim=1)
        
    def forward(self, x):
        x = self.conv(x)
        return self.activation(x)

class SwiGLULinear(nn.Module):
    """
    Linear layer with SwiGLU activation
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        # Double the output features since SwiGLU will halve them
        self.linear = nn.Linear(in_features, out_features * 2)
        # Use SwiGLU module with dim=-1 for Linear outputs (feature dimension)
        self.activation = SwiGLU(dim=-1)
        
    def forward(self, x):
        x = self.linear(x)
        return self.activation(x)

class ResidualBlock(nn.Module):
    """
    Residual block with SwiGLU activation
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = SwiGLUConv2d(
            channels, 
            channels, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        
    def forward(self, x):
        return x + self.conv(x)

class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor with SwiGLU activations and residual blocks
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim=512):
        super().__init__(observation_space, features_dim)
        
        # Initial convolution to adjust channels
        self.initial_conv = SwiGLUConv2d(
            observation_space.shape[0],  # Input channels (usually 3 for RGB)
            64,                         # Output channels
            kernel_size=7,              # Larger initial kernel
            stride=2,                   # Reduce spatial dimensions
            padding=3                   # Preserve spatial dimensions
        )
        
        # Residual blocks
        self.cnn = nn.Sequential(
            ResidualBlock(64),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduce spatial dimensions
            ResidualBlock(64),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduce spatial dimensions
            ResidualBlock(64),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduce spatial dimensions
            ResidualBlock(64),

            nn.Flatten()
        )
        
        # Compute the flattened size after CNN by passing a dummy tensor
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            cnn_output = self.cnn(self.initial_conv(sample))
            n_flatten = cnn_output.shape[1]
            print(f"Flattened CNN output size: {n_flatten}")
            
        # Final linear layer with SwiGLU activation
        self.fc = SwiGLULinear(n_flatten, features_dim)
        
    def forward(self, observations):
        x = self.initial_conv(observations)
        x = self.cnn(x)
        x = self.fc(x)
        return x