from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel
import torch

# ======== Parameters ========
model_id = "google/gemma-3-4b-pt"   # Base model ID
ft_model_path = "./google/gemma-3-4b-pt-bible"  # Path to fine-tuned model
use_quant = False  # Set to True for 8-bit inference, False for full precision
device = "cuda" if torch.cuda.is_available() else "cpu"

# ======== Load the Tokenizer ========
if "gemma" in model_id:
    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = processor.tokenizer
else:
    tokenizer = AutoTokenizer.from_pretrained(model_id)

# ======== Load the Base Model with Adapters ========
if use_quant:
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16,
        bnb_8bit_use_double_quant=False,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        ft_model_path,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
else:
    base_model = AutoModelForCausalLM.from_pretrained(
        ft_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )

# ======== Load the LoRA Adapters ========
# Merge the LoRA adapters into the base model
# model = PeftModel.from_pretrained(base_model, ft_model_path)
# model = model.merge_and_unload()  # Merge LoRA adapters into the model for inference
base_model.eval()

# ======== Inference ========
prompt = "The Divine Council is"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# Configure generation parameters
generation_config = GenerationConfig(
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95,
    top_k=50,
    repetition_penalty=1.2,
    do_sample=True,
)

# Generate text
with torch.no_grad():
    outputs = base_model.generate(
        input_ids=input_ids,
        generation_config=generation_config
    )

# Decode and print
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\nGenerated Text:\n", decoded_output)
