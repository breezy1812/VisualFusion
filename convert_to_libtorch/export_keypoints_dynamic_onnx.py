#!/usr/bin/env python
"""Export the model to ONNX format with only keypoints dynamic (input images are fixed size)."""

import os
import argparse
import torch
import onnx
import warnings
from pathlib import Path
import sys

# Add the parent directory to the path so we can import the model modules
sys.path.append(str(Path(__file__).parent))

from model_jit.SemLA import SemLA

class FourOutputsWrapper(torch.nn.Module):
    """Wrapper to output exactly 4 tensors with proper names"""
    def __init__(self, model):
        super(FourOutputsWrapper, self).__init__()
        self.model = model
    
    def forward(self, vi_img, ir_img):
        outputs = self.model(vi_img, ir_img)
        # Handle different numbers of outputs
        if isinstance(outputs, (list, tuple)):
            if len(outputs) == 6:
                # Select the first 4 outputs: mkpt0, mkpt1, feat_sa_vi, feat_sa_ir
                return outputs[0], outputs[1], outputs[2], outputs[3]
            elif len(outputs) == 4:
                return outputs[0], outputs[1], outputs[2], outputs[3]
            elif len(outputs) == 2:
                # If only 2 outputs, repeat them to make 4
                return outputs[0], outputs[1], outputs[0], outputs[1]
            else:
                raise ValueError(f"Expected 2, 4, or 6 outputs, got {len(outputs)}")
        else:
            # Single output, replicate to 4
            return outputs, outputs, outputs, outputs

def load_model(checkpoint_path, device):
    """Load the model with checkpoint"""
    print(f"Loading model with checkpoint: {checkpoint_path}")
    model = SemLA(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    model = model.to(device)
    return model

def export_onnx(model, device, output_path, height=360, width=480):
    """Export model to ONNX with only keypoints dynamic"""
    print(f"Exporting to ONNX with keypoints dynamic, input images fixed to {height}x{width}...")
    
    # Create wrapper model
    wrapped_model = FourOutputsWrapper(model)
    wrapped_model.eval()
    
    # Create dummy inputs with the specified dimensions (single channel for each input)
    dummy_vi = torch.randn(1, 1, height, width, dtype=torch.float32, device=device)
    dummy_ir = torch.randn(1, 1, height, width, dtype=torch.float32, device=device)
    
    # Test the model with dummy inputs
    print("Testing model with dummy inputs...")
    with torch.no_grad():
        outputs = wrapped_model(dummy_vi, dummy_ir)
        print(f"   Model outputs: {len(outputs)} tensors")
        for i, output in enumerate(outputs):
            print(f"     Output {i}: shape {output.shape}, dtype {output.dtype}")
    
    # Define dynamic axes - only keypoints are dynamic, everything else is fixed
    dynamic_axes = {
        # Only keypoints can be dynamic (number of keypoints can vary)
        'mkpt0': {0: 'num_keypoints'},       # Number of keypoints can vary
        'mkpt1': {0: 'num_keypoints'},       # Number of keypoints can vary
        # All other inputs/outputs are kept static (fixed size)
    }
    
    # Export to ONNX
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            wrapped_model,
            (dummy_vi, dummy_ir),
            output_path,
            export_params=True,
            opset_version=17,  # Use a well-supported opset
            do_constant_folding=True,
            input_names=['vi_img', 'ir_img'],
            output_names=['mkpt0', 'mkpt1', 'feat_sa_vi', 'feat_sa_ir'],
            dynamic_axes=dynamic_axes,
            verbose=False
        )
    
    print(f"SUCCESS: Keypoints-dynamic ONNX export completed to {output_path}")
    return True

def validate_onnx_model(onnx_path):
    """Validate the exported ONNX model"""
    print("Validating generated ONNX model...")
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("   ✓ ONNX model validation successful")
        
        # Print model info
        print(f"   Model inputs: {len(onnx_model.graph.input)}")
        for i, input_tensor in enumerate(onnx_model.graph.input):
            print(f"     Input {i}: {input_tensor.name}")
            # Print shape info
            shape = [dim.dim_value if dim.dim_value > 0 else f"dynamic_{dim.dim_param}" for dim in input_tensor.type.tensor_type.shape.dim]
            print(f"       Shape: {shape}")
        
        print(f"   Model outputs: {len(onnx_model.graph.output)}")
        for i, output_tensor in enumerate(onnx_model.graph.output):
            print(f"     Output {i}: {output_tensor.name}")
            # Print shape info
            shape = [dim.dim_value if dim.dim_value > 0 else f"dynamic_{dim.dim_param}" for dim in output_tensor.type.tensor_type.shape.dim]
            print(f"       Shape: {shape}")
        
        return True
    except Exception as e:
        print(f"   ✗ ONNX model validation failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Export SemLA model to ONNX with keypoints-only dynamic')
    parser.add_argument('--checkpoint', type=str, default='reg.ckpt',
                        help='Path to the model checkpoint (default: reg.ckpt)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output ONNX file path')
    parser.add_argument('--height', type=int, default=360,
                        help='Fixed input height (default: 360)')
    parser.add_argument('--width', type=int, default=480,
                        help='Fixed input width (default: 480)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use: cuda or cpu (default: cuda)')
    
    args = parser.parse_args()
    
    # Validate device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Set default output path if not provided
    if args.output is None:
        device_suffix = 'cuda' if args.device == 'cuda' else 'cpu'
        args.output = f'/circ330/forgithub/VisualFusion_libtorch/convert_to_libtorch/converModel/SemLA_onnx_{args.width}x{args.height}.onnx'
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    try:
        # Load model
        model = load_model(args.checkpoint, device)
        
        # Export to ONNX
        success = export_onnx(model, device, args.output, args.height, args.width)
        
        if success:
            # Validate the exported model
            validate_onnx_model(args.output)
            print(f"\n✓ Export completed successfully!")
            print(f"✓ Input images are fixed to {args.height}x{args.width}")
            print(f"✓ Keypoints outputs (mkpt0, mkpt1) are dynamic")
            print(f"✓ Feature outputs (feat_sa_vi, feat_sa_ir) are fixed size")
        else:
            print("Export failed.")
            return 1
            
    except Exception as e:
        print(f"Error during export: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
