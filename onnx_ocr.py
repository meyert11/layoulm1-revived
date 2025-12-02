from rapidocr_onnxruntime import RapidOCR

# Initialize with your custom model paths
engine = RapidOCR(
    # 1. Detection Model
    det_model_path='models/ch_PP-OCRv4_det_infer.onnx',
    
    # 2. Recognition Model (and its keys file)
    rec_model_path='models/ch_PP-OCRv4_rec_infer.onnx',
    rec_keys_path='models/ppocr_keys_v1.txt',  # Crucial for custom rec models
    
    # 3. Classification Model (Optional, set use_cls=True if using)
    cls_model_path='models/ch_ppocr_mobile_v2.0_cls_infer.onnx',
    
    # 4. GPU/Execution Configuration
    use_det=True,
    use_rec=True,
    use_cls=False,            # Set to True if you provided a cls model
    use_cuda=True,            # Enable GPU
    det_use_cuda=True,        # Specific flags may be needed in some versions
    rec_use_cuda=True,
    cls_use_cuda=True
)

# Run inference
img_path = 'test_image.jpg'
result, elapse = engine(img_path)

print(result)
