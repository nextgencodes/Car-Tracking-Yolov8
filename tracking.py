from ultralytics import YOLO

# Load an official or custom model
model = YOLO('yolov8n.pt')  # Load an official Detect model
#model = YOLO('yolov8n-seg.pt')  # Load an official Segment model
#model = YOLO('yolov8n-pose.pt')  # Load an official Pose model
#model = YOLO('path/to/best.pt')  # Load a custom trained model

# Perform tracking with the model
results = model.track(source="https://youtu.be/mwN6l3O1MNI", show=True, device='cpu', conf=0.4, iou=0.5)  # Tracking with default tracker
#results = model.track(source="Data/Example Videos/test1.mp4", show=True, device='cpu')  # Tracking with default tracker


'''[CPU, QuantizedCPU, BackendSelect, Python, FuncTorchDynamicLayerBackMode, Functionalize, Named, Conjugate, 
Negative, ZeroTensor, ADInplaceOrView, AutogradOther, AutogradCPU, AutogradCUDA, AutogradXLA,
 AutogradMPS, AutogradXPU, AutogradHPU, AutogradLazy, AutogradMeta, Tracer, AutocastCPU, AutocastCUDA,
 FuncTorchBatched, FuncTorchVmapMode, Batched, VmapMode, FuncTorchGradWrapper, PythonTLSSnapshot, 
 FuncTorchDynamicLayerFrontMode, PreDispatch, PythonDispatcher]'''