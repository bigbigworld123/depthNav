import torch
import torch.nn as nn
from typing import Dict, Any
import numpy as np

def count_parameters(model: nn.Module) -> int:
    """
    Count total number of parameters in a PyTorch model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_flops(model: nn.Module, input_tensor: Dict[str, torch.Tensor]) -> int:
    """
    Calculate approximate FLOPs for a forward pass of the model
    """
    # This is a simplified estimation - for more accurate FLOPs calculation,
    # you might want to use libraries like thop or fvcore
    total_flops = 0
    
    # For feature extractor (approximation)
    if hasattr(model, 'feature_extractor'):
        # This is a rough estimation based on typical operations
        # You would need to implement detailed FLOPs calculation for each layer
        features_dim = model.feature_extractor.features_dim
        total_flops += features_dim * 10  # Rough estimate for feature extraction
    
    # For recurrent layers
    if hasattr(model, 'recurrent_extractor'):
        if hasattr(model.recurrent_extractor, 'hidden_size'):
            hidden_size = model.recurrent_extractor.hidden_size
            # GRU FLOPs approximation
            total_flops += 3 * (features_dim * hidden_size + hidden_size * hidden_size) * 2
    
    # For policy network (MLP)
    if hasattr(model, 'policy_net'):
        for layer in model.policy_net:
            if isinstance(layer, nn.Linear):
                in_features = layer.in_features
                out_features = layer.out_features
                # Each neuron: in_features multiplications + (in_features-1) additions
                total_flops += out_features * (in_features * 2 - 1)
    
    return total_flops

def create_sample_observation(observation_space):
    """
    Create sample observations for model input
    """
    # Create sample observations matching the observation space
    obs = {}
    for key, space in observation_space.items():
        if hasattr(space, 'shape'):
            obs[key] = torch.randn(1, *space.shape)
        else:
            # Handle other space types as needed
            obs[key] = torch.randn(1, 10)  # Default size
    return obs


def main():
    # We need to import the required modules
    try:
        from depthnav.policies.multi_input_policy import MultiInputPolicy
        from depthnav.policies.extractors import StateTargetImageExtractor
        from gymnasium import spaces
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("Make sure you have installed the package with 'pip install -e .'")
        return

    # Example configuration for a typical DepthNav model
    # You would need to adjust these based on your actual model configuration
    
    # Define observation space (example)
    observation_space = spaces.Dict({
        "state": spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32),
        "image": spaces.Box(low=0, high=255, shape=(64, 64, 2), dtype=np.uint8),
        "target": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
    })
    
    # Define network architecture
    net_arch = {
        "recurrent": {
            "class": "GRUCell",
            "kwargs": {"hidden_size": 128}
        },
        "mlp_layer": [128, 64, 32, 4]
    }
    
    # Create model
    model = MultiInputPolicy(
        observation_space=observation_space,
        net_arch=net_arch,
        activation_fn="relu",
        output_activation_fn="identity",
        feature_extractor_class=StateTargetImageExtractor,
        feature_extractor_kwargs={},
        device="cpu"
    )
    
    print("Model Architecture:")
    print(model)
    print("\n" + "="*50)
    
    # Count parameters
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Create sample input
    sample_obs = create_sample_observation(observation_space)
    
    # Calculate FLOPs (approximate)
    flops = calculate_flops(model, sample_obs)
    print(f"Approximate FLOPs per forward pass: {flops:,}")
    
    # More detailed parameter breakdown
    print("\n" + "="*50)
    print("Parameter breakdown by module:")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if params > 0:
            print(f"{name}: {params:,} parameters")
    
    # Feature extractor details
    if hasattr(model, 'feature_extractor'):
        print(f"\nFeature extractor output dimension: {model.feature_extractor.features_dim}")
    
    if hasattr(model, 'recurrent_extractor'):
        if hasattr(model.recurrent_extractor, 'hidden_size'):
            print(f"Recurrent layer hidden size: {model.recurrent_extractor.hidden_size}")


if __name__ == "__main__":
    main()