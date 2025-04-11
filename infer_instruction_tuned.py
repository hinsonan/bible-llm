from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from jinja2 import Template
from peft import PeftModel

# Set the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load your fine-tuned model and tokenizer (make sure the model path is correct)
model_id = "google/gemma-3-4b-pt"
model_path = "./google/gemma-3-4b-pt-bible-it"  # Path where the fine-tuned model was saved
tokenizer = AutoTokenizer.from_pretrained(model_path)
# Load base model
model = AutoModelForCausalLM.from_pretrained(model_path)

# Load LoRA adapters
model = PeftModel.from_pretrained(model, model_path)

# Merge LoRA adapters for efficiency
model = model.merge_and_unload()
model.to(device)
model.eval()

# Your chat template (same as used during training)
chat_template = """
{%- if messages[0]['role'] == 'system' -%}
    {%- if messages[0]['content'] is string -%}
        {%- set first_user_prefix = messages[0]['content'] + '\n' -%}
    {%- else -%}
        {%- set first_user_prefix = messages[0]['content'][0]['text'] + '\n' -%}
    {%- endif -%}
    {%- set loop_messages = messages[1:] -%}
{%- else -%}
    {%- set first_user_prefix = "" -%}
    {%- set loop_messages = messages -%}
{%- endif -%}
{%- for message in loop_messages -%}
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}
        {{ raise_exception("Conversation roles must alternate user/assistant/user/assistant/...") }}
    {%- endif -%}
    {%- if (message['role'] == 'assistant') -%}
        {%- set role = "model" -%}
    {%- else -%}
        {%- set role = message['role'] -%}
    {%- endif -%}
    {{ '<start_of_turn>' + role + '\n' + (first_user_prefix if loop.first else "") }}
    {%- if message['content'] is string -%}
        {{ message['content'] | trim }}
    {%- elif message['content'] is iterable -%}
        {%- for item in message['content'] -%}
            {%- if item['type'] == 'image' -%}
                {{ '<start_of_image>' }}
            {%- elif item['type'] == 'text' -%}
                {{ item['text'] | trim }}
            {%- endif -%}
        {%- endfor -%}
    {%- else -%}
        {{ raise_exception("Invalid content type") }}
    {%- endif -%}
    {{ '<end_of_turn>\n' }}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{'<start_of_turn>model\n'}}
{%- endif -%}
"""

# Create a Jinja2 Template from your chat template
template = Template(chat_template)

# Define your conversation input (modify as needed)
chat = {
    "messages": [
        {"role": "system", "content": 'You are a helpful and respectful assistant with deep knowledge of the Bible, Christian theology, history, and traditions. You answer questions clearly and compassionately, citing Scripture when appropriate and remaining sensitive to different Christian perspectives. When possible, provide references (e.g., book, chapter, and verse) to support your responses. If a question is theological or interpretive, acknowledge differing views graciously and stay grounded in biblical context. Your goal is to inform, guide, and encourage users with wisdom and humility.'},
        {"role": "user", "content": "where in the bible does it mention nephilem"}
    ]
}

# Render the conversation using the template. 
# The flag `add_generation_prompt` signals that a generation turn should be appended.
formatted_input = template.render(messages=chat["messages"], add_generation_prompt=True)

print("Formatted input:")
print(formatted_input)

# Tokenize the formatted input text
inputs = tokenizer(formatted_input, return_tensors="pt", padding=True, truncation=False).to(device)

# Generate the response
outputs = model.generate(
    **inputs,
    max_new_tokens=512,          # Change as needed
    pad_token_id=tokenizer.eos_token_id,
    do_sample=True,              # For sampling-based generation; disable if using greedy/beam search
    temperature=0.7,
    top_p=0.9,
    top_k=100,
    repetition_penalty=1.2,
)

# outputs = model.generate(
#     **inputs,
#     max_new_tokens=512,
#     num_beams=5,           # Number of beams to track
#     do_sample=False,       # Disable sampling for pure beam search
#     early_stopping=True,   # Stop when all beams reach EOS
#     no_repeat_ngram_size=2, # Avoid repeating bigrams
#     top_k=None,
#     top_p=None,
# )

# outputs = model.generate(
#     **inputs,
#     max_new_tokens=512,
#     num_beams=1,           # Number of beams to track
#     do_sample=True,       
#     early_stopping=True,   # Stop when all beams reach EOS
#     top_k=64,
#     top_p=0.9,
#     temperature=0.7
# )

# Decode the generated tokens to text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\nGenerated output:")
print(generated_text)
