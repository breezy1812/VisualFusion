import torch
import torch.nn as nn
from model_jit.SemLA import SemLA

# Create a wrapper that eliminates dynamic shapes for ONNX export
class SemLA_ONNX_Wrapper(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model
        
    def forward(self, img_vi, img_ir):
        # Get the original outputs
        mkpts0, mkpts1, feat_sa_vi, feat_sa_ir, sa_ir, score = self.model(img_vi, img_ir)
        
        # For ONNX compatibility, we'll return fixed-size tensors
        # Pad or truncate to a fixed size (e.g., max 500 keypoints)
        max_points = 500
        
        # Create fixed-size outputs
        if mkpts0.shape[0] > max_points:
            mkpts0_fixed = mkpts0[:max_points]
            mkpts1_fixed = mkpts1[:max_points]
        else:
            # Pad with zeros if fewer points
            pad_size = max_points - mkpts0.shape[0]
            mkpts0_fixed = torch.cat([mkpts0, torch.zeros(pad_size, 2, device=mkpts0.device)], dim=0)
            mkpts1_fixed = torch.cat([mkpts1, torch.zeros(pad_size, 2, device=mkpts1.device)], dim=0)
        
        return mkpts0_fixed, mkpts1_fixed, feat_sa_vi, feat_sa_ir, sa_ir

def main():
    device = torch.device("cpu")
    fpMode = torch.float32
    
    # Load the original model
    matcher = SemLA(device=device, fp=fpMode)
    matcher.load_state_dict(torch.load(f"./reg.ckpt"), strict=False)
    matcher = matcher.eval().to(device, dtype=fpMode)
    
    # Create the wrapper
    wrapper_model = SemLA_ONNX_Wrapper(matcher)
    wrapper_model.eval()
    
    width = 320
    height = 240
    
    # Create sample inputs
    torch_input_1 = torch.randn(1, 1, height, width).to(device)
    torch_input_2 = torch.randn(1, 1, height, width).to(device)
    
    # First try with torch.jit.trace to handle dynamic shapes
    print("Tracing model...")
    try:
        traced_model = torch.jit.trace(wrapper_model, (torch_input_1, torch_input_2))
        print("Model traced successfully!")
        
        # Save traced model
        traced_model.save(f"/circ330/forgithub/VisualFusion_libtorch/convert_to_libtorch/converModel/SemLA_traced_{width}x{height}.pt")
        print("Traced model saved!")
        
    except Exception as e:
        print(f"Tracing failed: {e}")
        return
    
    # Now export to ONNX from the traced model
    print("Exporting to ONNX...")
    try:
        torch.onnx.export(
            traced_model,
            (torch_input_1, torch_input_2),
            f"/circ330/forgithub/VisualFusion_libtorch/convert_to_libtorch/converModel/SemLA_traced_opset10_{width}x{height}.onnx",
            verbose=True,
            opset_version=10,
            input_names=["vi_img", "ir_img"],
            output_names=["mkpt0", "mkpt1", "feat_sa_vi", "feat_sa_ir", "sa_ir"],
            # Remove dynamic axes for better compatibility
            # dynamic_axes=None
        )
        print("ONNX export successful!")
        
    except Exception as e:
        print(f"ONNX export failed: {e}")
        # Try with simpler approach - only export the backbone
        try:
            print("Trying to export only backbone...")
            backbone_model = matcher.backbone
            torch.onnx.export(
                backbone_model,
                torch.cat((torch_input_1, torch_input_2), dim=0),
                f"/circ330/forgithub/VisualFusion_libtorch/convert_to_libtorch/converModel/SemLA_backbone_opset10_{width}x{height}.onnx",
                verbose=True,
                opset_version=10,
                input_names=["input"],
                output_names=["feat_reg_vi", "feat_reg_ir", "feat_sa_vi", "feat_sa_ir"]
            )
            print("Backbone ONNX export successful!")
        except Exception as e2:
            print(f"Backbone export also failed: {e2}")

if __name__ == "__main__":
    main()
