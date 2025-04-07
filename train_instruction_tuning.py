from transformers import AutoProcessor, AutoModelForCausalLM, TrainingArguments, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
from peft import PeftModel, LoraConfig, get_peft_model
from datasets import load_dataset
from jinja2 import Template
import torch

# Your template (Jinja syntax)
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

def format_chat(example):
    """
    Formats the dataset into a consistent chat structure with:
    - A system message: "You are helpful"
    - User and assistant message pairs.
    """
    
    # Add a default system message
    system_message = {"role": "system", "content": "You are a helpful assistant"}
    
    # Initialize messages list with the system message
    messages = [system_message]

    # Add user and assistant messages
    if "messages" not in example:
        # Handle different dataset formats
        if "instruction" in example:
            messages.append({"role": "user", "content": example["instruction"]})
        elif "input" in example:
            messages.append({"role": "user", "content": example["input"]})

        if "output" in example:
            messages.append({"role": "assistant", "content": example["output"]})
        elif "response" in example:
            messages.append({"role": "assistant", "content": example["response"]})
        elif "demonstration" in example:
            messages.append({"role": "assistant", "content": example["demonstration"]})
    else:
        # If the dataset already contains messages, just prepend the system message
        messages.extend(example["messages"])

    # Return the formatted chat structure
    return {"messages": messages}


def tokenize_function(batch):
    """
    Tokenizes a batch of messages, handling both single and multi-part content.
    """
    texts = []

    # Iterate over the batch properly
    for example in batch["messages"]:
        # Each example contains a list of message dictionaries
        conversation = []
        
        for message in example:  # Iterate over messages in a conversation
            if isinstance(message, dict):  # Ensure it's a dictionary
                role = message.get("role", "user")
                content = message.get("content", "")

                # Handle multi-part content
                if isinstance(content, list):
                    content = " ".join(
                        [item.get("text", "") for item in content if isinstance(item, dict)]
                    )

                conversation.append(f"<{role}>: {content}")

        # Join conversation into a single string
        texts.append(" ".join(conversation))

    # Tokenize the batch of conversation texts
    tokens = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    # Add labels (CLM task requires labels to be input_ids shifted by one)
    labels = tokens["input_ids"].clone()

    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "labels": labels
    }


if __name__ == "__main__":
    model_id = "google/gemma-3-4b-pt"   # Base model ID
    ft_model_path = "./google/gemma-3-4b-pt-bible"  # Path to fine-tuned model
    preprocess_data = False
    use_quant = True  # Set to True for 8-bit inference, False for full precision
    use_lora = True
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
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )

    # ======== Load the LoRA Adapters ========
    # Merge the LoRA adapters into the base model
    model = PeftModel.from_pretrained(model, ft_model_path)
    model = model.merge_and_unload()
    for param in model.parameters():
        param.requires_grad = True

    # Set up new LoRA adapters for instruction tuning if needed
    if use_lora:
        print("Setting up LoRA for instruction tuning...")
        # Prepare model for k-bit training if using quantization
        # Configure LoRA parameters
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear"
        )
        
        # Apply LoRA to the model
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    dataset = load_dataset('HuggingFaceH4/helpful-instructions')
    dataset = dataset.map(format_chat)
    if preprocess_data:
        dataset = dataset.map(tokenize_function, batched=True)
        dataset = dataset.remove_columns([col for col in dataset['train'].column_names if col not in ["input_ids", "attention_mask", "labels"]])
    else:
        dataset = dataset.remove_columns([col for col in dataset['train'].column_names if col not in ["messages"]])

    print(dataset['train'])

    tokenizer.chat_template = chat_template
    
    # Set up SFT config
    sft_config = SFTConfig(
        max_seq_length=268,
        packing=True,
        # dataset_text_field="messages",
        output_dir="./gemma-3-4b-instruction-tuned-bible",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        save_total_limit=3,
        eval_strategy="no",
        save_strategy="epoch",
        remove_unused_columns=False,
        bf16=torch.cuda.is_available(),
        # load_best_model_at_end=True,
        optim="adamw_torch",
        label_names=['labels']
    )
    
    # Initialize SFT Trainer
    print("Initializing SFT Trainer...")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if "validation" in dataset else dataset["test"] if "test" in dataset else None,
        peft_config=lora_config if use_lora else None,
        processing_class=tokenizer,
        
    )
    
    # Train the model
    print("Starting instruction tuning...")
    trainer.train()

    model.save_pretrained(f"./{model_id}-bible-it")