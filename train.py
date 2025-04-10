from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    AutoTokenizer,
)
from datasets import load_from_disk
import torch
from peft import get_peft_model, LoraConfig


def tokenize_function(examples):
    encoding = tokenizer(
        text=examples["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=50
    )
    encoding["labels"] = encoding["input_ids"].copy()
    return encoding


if __name__ == "__main__":

    dataset = load_from_disk("./data/hugging_face_bible_dataset_format")["train"]
    model_id = "google/gemma-3-4b-pt"
    # model_id = "google/gemma-3-1b-pt"
    # model_id = "meta-llama/Llama-3.2-1B"

    use_quant=False
    use_lora=False

    if "Llama" in model_id:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
    elif "google/gemma-3-4b-pt" in model_id:
        processor = AutoProcessor.from_pretrained(model_id)
        tokenizer = processor.tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    tokenized_datasets = tokenized_datasets.remove_columns(["reference", "text"])

    split_dataset = tokenized_datasets.train_test_split(test_size=0.1, seed=42)

    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    if use_quant:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,  # Keep bfloat16 as it works well with Gemma
            bnb_8bit_use_double_quant=False,  # Avoid double quantization for better stability
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            attn_implementation="eager",
            torch_dtype=torch.bfloat16
        )

    if use_lora:
        lora_config = LoraConfig(
            r=8,  # Low-rank adapter dimension
            lora_alpha=32,  # Scaling factor for the LoRA layers
            lora_dropout=0.1,  # Dropout rate for LoRA layers
            target_modules="all-linear",  # Which layers to apply LoRA to
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
        output_dir=f"./{model_id}-bible",  # Where to save the model
        eval_strategy="epoch",  # Evaluate at the end of each epoch
        save_strategy="epoch",  # Save the model at the end of each epoch
        learning_rate=1e-5,  # Learning rate for fine-tuning
        per_device_train_batch_size=4,  # Batch size per device during training
        per_device_eval_batch_size=4,  # Batch size per device during evaluation
        num_train_epochs=3,  # Number of training epochs
        weight_decay=0.01,  # Weight decay to prevent overfitting
        logging_dir="./logs",  # Directory for storing logs
        logging_steps=10,  # Log every 10 steps
        save_steps=500,  # Save model every 500 steps
        load_best_model_at_end=True,  # Load the best model at the end of training
        push_to_hub=False,  # If you want to push the model to Hugging Face Hub
        label_names=["labels"],
        max_grad_norm=1.0,
        gradient_accumulation_steps=1,
        log_level="debug",
        bf16=True,
        optim="adamw_bnb_8bit"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    model.save_pretrained(f"./{model_id}-bible")
