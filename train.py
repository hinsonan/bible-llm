from transformers import AutoProcessor, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, BitsAndBytesConfig, AutoTokenizer
from datasets import load_from_disk
from PIL import Image
import requests
import torch
from peft import get_peft_model, LoraConfig

def tokenize_function(examples):
    valid_texts = [text for text in examples["text"] if text and text.strip()]
    # Ensure the labels are added here for training
    encoding = tokenizer(text=valid_texts, truncation=True, padding="max_length", max_length=50)
    encoding["labels"] = encoding["input_ids"].copy()  # The labels are same as input_ids for causal language modeling
    return encoding

if __name__ == "__main__":

    dataset = load_from_disk("./data/hugging_face_bible_dataset_format")
    model_id = "google/gemma-3-1b-pt"
    # model_id = "meta-llama/Llama-3.2-1B"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if 'Llama' in model_id:
        tokenizer.pad_token = tokenizer.eos_token
    # tokenizer = processor.tokenizer

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    tokenized_datasets = tokenized_datasets.remove_columns(["reference", "text"])

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16,  # Keep bfloat16 as it works well with Gemma
        bnb_8bit_use_double_quant=False  # Avoid double quantization for better stability
    )

    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto", attn_implementation='eager', torch_dtype=torch.bfloat16)

    lora_config = LoraConfig(
        r=8,  # Low-rank adapter dimension
        lora_alpha=32,  # Scaling factor for the LoRA layers
        lora_dropout=0.1,  # Dropout rate for LoRA layers
        target_modules='all-linear',  # Which layers to apply LoRA to
        bias="none",  # Whether to use biases in LoRA layers
        task_type="CAUSAL_LM",  # Task type for causal language modeling
    )

    # Attach LoRA adapters to the model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()


    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=f"./{model_id}-bible",         # Where to save the model
        eval_strategy="epoch",       # Evaluate at the end of each epoch
        save_strategy="epoch",             # Save the model at the end of each epoch
        learning_rate=1e-5,                # Learning rate for fine-tuning
        per_device_train_batch_size=4,     # Batch size per device during training
        per_device_eval_batch_size=4,      # Batch size per device during evaluation
        num_train_epochs=3,                # Number of training epochs
        weight_decay=0.01,                 # Weight decay to prevent overfitting
        logging_dir="./logs",              # Directory for storing logs
        logging_steps=10,                  # Log every 10 steps
        save_steps=500,                    # Save model every 500 steps
        load_best_model_at_end=True,       # Load the best model at the end of training
        push_to_hub=False,                 # If you want to push the model to Hugging Face Hub
        label_names=["labels"],
        max_grad_norm=1.0,
        gradient_accumulation_steps=1,
        log_level="debug"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["train"],  # You can separate train/eval if you have validation data
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    model.save_pretrained(f"./{model_id}-bible")