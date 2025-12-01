# Cell 1: Install dependencies with CUDA support
# NOTE: If on Windows (PowerShell), replace 'CMAKE_ARGS...' with: $env:CMAKE_ARGS="-DGGML_CUDA=on"; pip install llama-cpp-python
!CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python huggingface_hub

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
