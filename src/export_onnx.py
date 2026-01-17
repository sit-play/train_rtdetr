from ultralytics import RTDETR
import torch
import onnx

print("="*50)
print("Testing RT-DETR Export (opset 16)")
print("="*50)

# Load your custom trained model
model_path = "model/runs/segment/Rhizoctonia.v33/weights/best.pt"
print(f"\nLoading model from: {model_path}")

try:
    model = RTDETR(model_path)
    print("✅ Model loaded successfully!")
    
    # Export with opset 16
    print("\nExporting to ONNX (opset=16)...")
    model.export(format="onnx", imgsz=640, simplify=True, opset=16)
    print("✅ RT-DETR ONNX export successful!")
    
    # Analyze ONNX model
    onnx_path = model_path.replace("best.pt", "best.onnx")
    print(f"\nLoading ONNX from: {onnx_path}")
    onnx_model = onnx.load(onnx_path)
    
    print("\n📊 Model Outputs:")
    for out in onnx_model.graph.output:
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"  {out.name}: {shape}")
    
    print("\n🔧 Ops used:")
    ops = set(n.op_type for n in onnx_model.graph.node)
    risky_ops = ['Attention', 'MultiHeadAttention', 'GridSample', 'NonZero', 'Where']
    print(f"Total unique ops: {len(ops)}")
    for op in sorted(ops):
        flag = "⚠️ " if op in risky_ops else ""
        print(f"  {flag}{op}")
    
    print("\n" + "="*50)
    print("Export completed successfully!")
    print("="*50)
    
except FileNotFoundError:
    print(f"❌ Error: Model file not found at {model_path}")
    print("Please check the path and try again.")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()