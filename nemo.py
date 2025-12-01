# Cell 1: Install dependencies with specific CUDA build (Cached)
import os
import sys

# 1. Setup Cache Settings
# We will store the built wheel in this directory so we don't have to compile C++ every time
wheel_cache_dir = "./llama_cpp_wheels"
os.makedirs(wheel_cache_dir, exist_ok=True)

# 2. Check for existing wheel to avoid rebuilding
# We look for any file in that folder that looks like the llama-cpp-python wheel
existing_wheels = [f for f in os.listdir(wheel_cache_dir) if "llama_cpp_python" in f and f.endswith(".whl")]

if existing_wheels:
    wheel_path = os.path.join(wheel_cache_dir, existing_wheels[0])
    print(f"Found cached GPU wheel: {existing_wheels[0]}")
    print("Installing from cache (Fast)...")
    !pip install "{wheel_path}"
else:
    print("No cached wheel found. Compiling from source with CUDA support (Slow - ~5-10 mins)...")
    print("This will be cached for future runs.")
    
    # Build the wheel specifically with CUDA enabled
    # NOTE: Windows users might need to set $env:CMAKE_ARGS="-DGGML_CUDA=on" in PowerShell context manually if this fails
    !CMAKE_ARGS="-DGGML_CUDA=on" pip wheel llama-cpp-python --wheel-dir="{wheel_cache_dir}"
    
    # Install the newly created wheel
    !pip install --find-links="{wheel_cache_dir}" llama-cpp-python

# 3. Install other dependencies
!pip install huggingface_hub

# Cell 2: Download Model & Initialize
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# 1. Download the specific quantization (Q4_K_M is the balance of speed/quality for 12B)
model_name = "bartowski/Mistral-Nemo-Instruct-2407-GGUF"
model_file = "Mistral-Nemo-Instruct-2407-Q4_K_M.gguf"

print(f"Downloading {model_file}...")
model_path = hf_hub_download(repo_id=model_name, filename=model_file)
print(f"Model downloaded to: {model_path}")

# 2. Load the model onto GPU
# n_gpu_layers=-1 moves ALL layers to VRAM.
# n_ctx=0 uses the model's default limit (128k).
# WARNING: 128k context requires massive VRAM. If you OOM, lower n_ctx to 32768 or 16384.
llm = Llama(
    model_path=model_path,
    n_gpu_layers=-1,      # -1 = Offload all layers to GPU
    n_ctx=32768,          # Start with 32k. Set to 0 for full 128k (may OOM without quantized cache)
    n_batch=512,          # Processing batch size
    verbose=True          # Set False to hide startup logs
)

print("Model Loaded successfully!")

# Cell 3: Define Helper Function & Run Test
def extract_data(markdown_text):
    """
    Sends markdown context to the model and asks for JSON extraction.
    """
    system_prompt = "You are a financial analyst. Extract PII and financial data into JSON format."
    
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract information from this text:\n\n{markdown_text}"}
        ],
        response_format={"type": "json_object"}, # Enforces valid JSON output
        temperature=0.1, # Low temp for factual extraction
        max_tokens=2000  # Reserve space for the response
    )
    
    return response['choices'][0]['message']['content']

# Test Payload
sample_markdown = """
# INVOICE #1024
**Date:** 2023-10-27
**Bill To:** John Doe (SSN: ***-**-1234)
**Amount:** $4,500.00
"""

print("Running extraction...")
result = extract_data(sample_markdown)
print("\n--- Extracted Data ---")
print(result)
