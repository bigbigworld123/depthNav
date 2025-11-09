#!/usr/bin/env python3

"""
Script to analyze a trained DepthNav model.
This script loads a trained model and computes its parameter count and FLOPs.
"""

import torch
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__)))

def count_parameters(model):
    """
    Count the total number of trainable parameters in a PyTorch model.
    
    Args:
        model (torch.nn.Module): The model to analyze
        
    Returns:
        int: Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_sample_input():
    """
    Create sample input tensors matching the DepthNav observation space.
    
    Returns:
        dict: Sample input tensors
    """
    # These dimensions match the typical DepthNav observation space
    sample_input = {
        "state": torch.randn(1, 10),      # State information (position, velocity, etc.)
        "target": torch.randn(1, 3),      # Target position
        "depth": torch.randn(1, 64, 64, 2)  # Depth images (2 channels)
    }
    return sample_input

def analyze_trained_model(model_path):
    """
    Analyze a trained DepthNav model.
    
    Args:
        model_path (str): Path to a trained model file
    """
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None
        
    try:
        # Import required modules
        import yaml
        from depthnav.policies.multi_input_policy import MultiInputPolicy
        from depthnav.policies.extractors import StateTargetImageExtractor
        from gymnasium import spaces
        
        print("=" * 60)
        print("Trained DepthNav Model Analysis")
        print("=" * 60)
        print(f"Model path: {model_path}")
        
        # Load the model checkpoint
        print("\nLoading model checkpoint...")
        checkpoint = torch.load(model_path, map_location='cpu')
        print("Checkpoint loaded successfully!")
        
        # Print checkpoint keys for debugging
        print(f"Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dict'}")
        
        # If checkpoint is a dict with 'policy_state_dict' key, use that
        if isinstance(checkpoint, dict) and 'policy_state_dict' in checkpoint:
            state_dict = checkpoint['policy_state_dict']
            print("Using 'policy_state_dict' from checkpoint")
        else:
            state_dict = checkpoint
            print("Using checkpoint directly as state_dict")
        
        # Print some information about the state dict
        print(f"State dict keys count: {len(state_dict.keys())}")
        print(f"Sample keys: {list(state_dict.keys())[:5]}")
        
        # Try to get model configuration from the checkpoint or use default
        # Based on the policy config file we found
        observation_space = spaces.Dict({
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32),
            "target": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "depth": spaces.Box(low=0, high=255, shape=(64, 64, 2), dtype=np.uint8)
        })
        
        # Network architecture based on the policy config
        net_arch = {
            "recurrent": {
                "class": "GRUCell",
                "kwargs": {"hidden_size": 192}
            },
            "mlp_layer": [4]
        }
        
        # Feature extractor configuration
        feature_extractor_kwargs = {
            "activation_fn": "leaky_relu",
            "net_arch": {
                "state": {
                    "mlp_layer": [192],
                    "bn": False,
                    "ln": True
                },
                "target": {
                    "mlp_layer": [192],
                    "bn": False,
                    "ln": True
                },
                "depth": {
                    "input_max_pool_H_W": [12, 16],
                    "kernel_size": [2, 3, 3],
                    "channels": [32, 64, 128],
                    "padding": [0, 0, 0],
                    "stride": [1, 1, 1],
                    "cnn_bn": False,
                    "mlp_layer": [192],
                    "bn": False,
                    "ln": True
                },
                "concatenate": False
            }
        }
        
        # Create the model
        print("\nCreating model architecture...")
        model = MultiInputPolicy(
            observation_space=observation_space,
            net_arch=net_arch,
            activation_fn="leaky_relu",
            output_activation_fn="identity",
            feature_extractor_class=StateTargetImageExtractor,
            feature_extractor_kwargs=feature_extractor_kwargs,
            device="cpu"
        )
        
        print("Model created successfully!")
        print(f"Model type: {type(model)}")
        
        # Count parameters before loading weights
        total_params_before = count_parameters(model)
        print(f"Total trainable parameters (before loading): {total_params_before:,}")
        
        # Try to load the state dict
        try:
            # Load the state dict
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            print(f"Model weights loaded with:")
            print(f"  - Missing keys: {len(missing_keys)}")
            print(f"  - Unexpected keys: {len(unexpected_keys)}")
            
            if missing_keys:
                print(f"Missing keys: {missing_keys[:5]}...")  # Show first 5
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5
                
        except Exception as e:
            print(f"Error loading model weights: {e}")
            print("Continuing with the model architecture analysis...")
        
        # Count parameters after loading weights (should be the same)
        total_params_after = count_parameters(model)
        print(f"Total trainable parameters (after loading): {total_params_after:,}")
        
        # Create sample input
        sample_input = create_sample_input()
        print("\nSample input created:")
        for key, value in sample_input.items():
            print(f"  {key}: {value.shape}")
        
        # Try to calculate FLOPs with thop
        try:
            from thop import profile
            print("\nCalculating FLOPs using thop...")
            flops, params = profile(model, inputs=(sample_input,))
            print(f"FLOPs (using thop): {flops:,.0f}")
            print(f"Parameters (using thop): {params:,}")
        except ImportError:
            print("\nthop library not found. Install it with: pip install thop")
            print("Approximate FLOPs calculation not implemented for this complex model.")
        
        # Detailed parameter breakdown
        print("\n" + "=" * 60)
        print("Parameter Breakdown by Module")
        print("=" * 60)
        for name, module in model.named_children():
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if params > 0:
                print(f"{name:25}: {params:>15,} parameters ({params/total_params_after*100:>6.2f}%)")
        
        # Additional model information
        print("\n" + "=" * 60)
        print("Additional Model Information")
        print("=" * 60)
        
        if hasattr(model, 'feature_extractor'):
            print(f"Feature extractor output dimension: {model.feature_extractor.features_dim}")
        
        if hasattr(model, 'recurrent_extractor'):
            if hasattr(model.recurrent_extractor, 'hidden_size'):
                print(f"Recurrent layer hidden size: {model.recurrent_extractor.hidden_size}")
        
        return model
        
    except Exception as e:
        print(f"Error analyzing model: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_trained_model.py <path_to_model.pth>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    print(f"Analyzing trained model: {model_path}")
    
    # Analyze the model
    model = analyze_trained_model(model_path)
    
    if model is not None:
        print("\nModel analysis completed successfully!")
    else:
        print("\nFailed to analyze the model.")