import os
import cv2
import numpy as np
import onnxruntime as ort
from pdf2image import convert_from_path
from rapidocr_onnxruntime import RapidOCR

# --- CONFIGURATION ---
DOCS_DIR = "/mnt/data/docs"
MODELS_DIR = "paddle_models"
DET_MODEL_PATH = os.path.join(MODELS_DIR, "det.onnx")
REC_MODEL_PATH = os.path.join(MODELS_DIR, "rec.onnx")
KEYS_PATH = os.path.join(MODELS_DIR, "keys.txt")

# The Fixed Width we will force on the Recognition Model
# 960 is wide enough for almost any text line.
STATIC_REC_WIDTH = 960 

# --- CRITICAL PATCHING SECTION ---

# 1. Save original methods
_original_init = ort.InferenceSession.__init__
_original_run = ort.InferenceSession.run

def patched_init(self, path_or_bytes, sess_options=None, providers=None, **kwargs):
    """
    Force MIGraphX Provider on ROCm 7.1.1
    """
    forced_providers = [('MIGraphXExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider']
    
    # Silence logs to reduce overhead
    if sess_options is None:
        sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3 
    
    # Identify which model this is
    model_name = os.path.basename(str(path_or_bytes))
    self._model_type = "REC" if "rec" in model_name.lower() else "OTHER"
    
    print(f"   >>> [Init] Loading {model_name} on MIGraphX (Type: {self._model_type})...")
    _original_init(self, path_or_bytes, sess_options, forced_providers, **kwargs)

def patched_run(self, output_names, input_feed, run_options=None):
    """
    Intercept Data -> Force Contiguous -> Force Static Shape (Padding)
    """
    new_feed = {}
    
    for name, tensor in input_feed.items():
        # A. MEMORY FIX: Force Contiguous (Fixes 'Param Type Mismatch' crash)
        if not tensor.flags['C_CONTIGUOUS']:
            tensor = np.ascontiguousarray(tensor)
            
        # B. SPEED FIX: Static Padding for Recognition Model
        # If this is the Rec model (Height is 32), pad width to STATIC_REC_WIDTH
        if self._model_type == "REC" and len(tensor.shape) == 4:
            b, c, h, w = tensor.shape
            
            # If it's smaller than our static target, PAD IT
            if w < STATIC_REC_WIDTH:
                # Create black canvas (0.0 or -0.5 depending on normalization, usually 0 works for padding)
                # Padding with 0 in the normalized float space is usually safe for OCR CTC decoders
                canvas = np.zeros((b, c, h, STATIC_REC_WIDTH), dtype=tensor.dtype)
                
                # Copy the real image into the left side
                canvas[:, :, :, :w] = tensor
                tensor = canvas
            
            # If it's larger (rare), we truncate or let it recompile (rare edge case)
            elif w > STATIC_REC_WIDTH:
                print(f"   [Warn] Text too long ({w}px), truncating to {STATIC_REC_WIDTH}...")
                tensor = tensor[:, :, :, :STATIC_REC_WIDTH]

        # Final safety check for contiguous after padding
        if not tensor.flags['C_CONTIGUOUS']:
             tensor = np.ascontiguousarray(tensor)
             
        new_feed[name] = tensor

    # Call the real GPU execution
    return _original_run(self, output_names, new_feed, run_options)

# Apply Patches
ort.InferenceSession.__init__ = patched_init
ort.InferenceSession.run = patched_run
# ----------------------------------------


def main():
    print("Initializing RapidOCR with Static Shape Patching...")
    
    try:
        # We DISABLE angle_cls because that model expects squares and breaks the logic
        engine = RapidOCR(
            det_model_path=DET_MODEL_PATH,
            rec_model_path=REC_MODEL_PATH,
            rec_keys_path=KEYS_PATH,
            use_det=True,
            use_rec=True,
            use_angle_cls=False, # Essential for stability
            use_gpu=True
        )
        print("✅ Engine Ready. (First run will take ~30s to compile the 960px kernel)")
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    if not os.path.exists(DOCS_DIR):
        print(f"Directory not found: {DOCS_DIR}")
        return

    pdf_files = [f for f in os.listdir(DOCS_DIR) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print("No PDFs found.")
        return
    
    target_pdf = os.path.join(DOCS_DIR, pdf_files[0])
    print(f"\nProcessing: {target_pdf}")
    print("=" * 50)

    try:
        pages = convert_from_path(target_pdf)
        print(f"Loaded {len(pages)} pages.")

        for i, page in enumerate(pages):
            print(f"\n--- Page {i+1} ---")
            img = np.array(page)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Timing
            t0 = cv2.getTickCount()
            result, _ = engine(img)
            t1 = cv2.getTickCount()
            time_s = (t1 - t0) / cv2.getTickFrequency()
            
            print(f"Inference Time: {time_s:.4f}s")
            
            if result:
                # Print first few results
                count = 0
                for item in result:
                    text = item[1]
                    conf = item[2]
                    if float(conf) > 0.6:
                        print(f"  > {text}")
                        count += 1
                        if count >= 5:
                            print("  ... (more)")
                            break
            else:
                print("  (No text found)")

    except Exception as e:
        print(f"Processing failed: {e}")

if __name__ == "__main__":
    main()
