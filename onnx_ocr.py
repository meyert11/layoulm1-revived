from rapidocr_onnxruntime import RapidOCR

# Initialize with your local models and force GPU (CUDA) execution
engine = RapidOCR(
    det_model_path='paddle_model/det.onnx',
    cls_model_path='paddle_model/cls.onnx',
    rec_model_path='paddle_model/ret.onnx', 
    rec_keys_path='paddle_model/key.txt',
    providers=['CUDAExecutionProvider'] 
)

# Test
img_path = 'test_image.jpg' 
result, elapse = engine(img_path)
print(result)
