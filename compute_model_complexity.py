#!/usr/bin/env python3

"""
Script to compute model complexity metrics including:
1. Total number of parameters
2. FLOPs (Floating Point Operations) using thop library if available
3. Approximate FLOPs calculation if thop is not available
"""

import torch
import numpy as np

def count_parameters(model):
    """
    Count the total number of trainable parameters in a PyTorch model.
    
    Args:
        model (torch.nn.Module): The model to analyze
        
    Returns:
        int: Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_flops_approximate(model, input_tensor):
    """
    Calculate approximate FLOPs for a forward pass of the model.
    This is a rough estimation and might not be completely accurate.
    
    Args:
        model (torch.nn.Module): The model to analyze
        input_tensor (dict): Sample input to the model
        
    Returns:
        int: Approximate number of FLOPs
    """
    total_flops = 0
    
    # For feature extractor (approximation)
    if hasattr(model, 'feature_extractor'):
        features_dim = model.feature_extractor.features_dim
        # Rough estimate for feature extraction (this is very approximate)
        total_flops += features_dim * 100
    
    # For recurrent layers
    if hasattr(model, 'recurrent_extractor'):
        if hasattr(model.recurrent_extractor, 'hidden_size'):
            hidden_size = model.recurrent_extractor.hidden_size
            features_dim = getattr(model.recurrent_extractor, 'input_size', features_dim)
            # GRU FLOPs approximation: 3 * (input_size * hidden_size + hidden_size * hidden_size) * 2
            total_flops += 3 * (features_dim * hidden_size + hidden_size * hidden_size) * 2
    
    # For policy network (MLP)
    if hasattr(model, 'policy_net'):
        for layer in model.policy_net:
            if isinstance(layer, torch.nn.Linear):
                in_features = layer.in_features
                out_features = layer.out_features
                # Each neuron: in_features multiplications + (in_features-1) additions
                total_flops += out_features * (in_features * 2 - 1)
    
    return total_flops

def calculate_flops_thop(model, input_tensor):
    """
    Calculate FLOPs using the thop library.
    
    Args:
        model (torch.nn.Module): The model to analyze
        input_tensor (dict): Sample input to the model
        
    Returns:
        tuple: (FLOPs, parameters) as reported by thop
    """
    try:
        from thop import profile
        # Prepare input for thop
        flops, params = profile(model, inputs=(input_tensor,))
        return flops, params
    except ImportError:
        print("thop library not found. Install it with: pip install thop")
        print("Using approximate FLOPs calculation instead.")
        return None, None

def create_sample_input(observation_space):
    """
    Create sample input tensors matching the observation space.
    
    Args:
        observation_space: The observation space of the model
        
    Returns:
        dict: Sample input tensors
    """
    obs = {}
    for key, space in observation_space.items():
        if hasattr(space, 'shape'):
            obs[key] = torch.randn(1, *space.shape)
        else:
            obs[key] = torch.randn(1, 10)  # Default size
    return obs

def analyze_model_complexity(model, observation_space):
    """
    Analyze the complexity of a model in terms of parameters and FLOPs.
    
    Args:
        model (torch.nn.Module): The model to analyze
        observation_space: The observation space of the model
    """
    print("=" * 60)
    print("Model Complexity Analysis")
    print("=" * 60)
    
    # Count parameters
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Create sample input
    sample_input = create_sample_input(observation_space)
    
    # Try to calculate FLOPs with thop
    flops_thop, params_thop = calculate_flops_thop(model, sample_input)
    
    if flops_thop is not None:
        print(f"FLOPs (using thop): {flops_thop:,.0f}")
        print(f"Parameters (using thop): {params_thop:,}")
    else:
        # Fallback to approximate calculation
        flops_approx = calculate_flops_approximate(model, sample_input)
        print(f"Approximate FLOPs: {flops_approx:,}")
    
    # Detailed parameter breakdown
    print("\n" + "=" * 60)
    print("Parameter Breakdown by Module")
    print("=" * 60)
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if params > 0:
            print(f"{name:25}: {params:>15,} parameters ({params/total_params*100:>6.2f}%)")
    
    # Additional model information
    print("\n" + "=" * 60)
    print("Additional Model Information")
    print("=" * 60)
    
    if hasattr(model, 'feature_extractor'):
        print(f"Feature extractor output dimension: {model.feature_extractor.features_dim}")
    
    if hasattr(model, 'recurrent_extractor'):
        if hasattr(model.recurrent_extractor, 'hidden_size'):
            print(f"Recurrent layer hidden size: {model.recurrent_extractor.hidden_size}")

# Example usage
if __name__ == "__main__":
    # Example usage with a simple model
    print("Example with a simple model:")
    
    # Create a simple model for demonstration
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(10, 64)
            self.linear2 = torch.nn.Linear(64, 32)
            self.linear3 = torch.nn.Linear(32, 1)
            self.relu = torch.nn.ReLU()
            
        def forward(self, x):
            x = self.relu(self.linear1(x))
            x = self.relu(self.linear2(x))
            x = self.linear3(x)
            return x
    
    model = SimpleModel()
    
    # Mock observation space
    class MockSpace:
        def __init__(self, shape):
            self.shape = shape
    
    observation_space = {"state": MockSpace((10,))}
    
    # Analyze the model
    analyze_model_complexity(model, observation_space)
    
    print("\nTo analyze your DepthNav model, import your model and pass it to analyze_model_complexity()")