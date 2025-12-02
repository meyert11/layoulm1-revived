import onnxruntime as ort
import numpy as np
import time

# 1. Enable Verbose Logging
sess_options = ort.SessionOptions()
sess_options.log_severity_level = 0  # 0 = Verbose (Prints EVERYTHING)

print("Attempting to load model onto GPU...")

try:
    # We load the detection model directly to test the GPU link
    # Adjust path if your paddle_model folder is elsewhere
    session = ort.InferenceSession(
        'paddle_model/det.onnx', 
        sess_options=sess_options, 
        providers=['CUDAExecutionProvider']
    )
    
    print("\n✅ Model loaded successfully!")
    print(f"Active Provider: {session.get_providers()[0]}")
    
    # 2. Quick Dummy Inference to check speed
    # Standard input size for detection model (1, 3, 640, 640)
    dummy_input = {session.get_inputs()[0].name: np.random.randn(1, 3, 640, 640).astype(np.float32)}
    
    start = time.time()
    session.run(None, dummy_input)
    print(f"Inference time: {time.time() - start:.4f}s")

except Exception as e:
    print("\n❌ MODEL LOAD FAILED")
    print("Read the error logs above carefully. Look for 'lib not found' or 'symbol lookup error'.")
    print(e)
