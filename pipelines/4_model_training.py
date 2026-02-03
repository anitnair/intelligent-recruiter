import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model

# ---------------------------
# Configuration
# ---------------------------
MODEL_ID = "google/gemma-300m"
TRAIN_FILE = "train.jsonl"
VAL_FILE = "val.jsonl"
OUTPUT_DIR = "./gemma300m-hr-copilot"

MAX_LENGTH = 512
BATCH_SIZE = 8
EPOCHS = 10
LR = 2e-4
PATIENCE = 4

# ---------------------------
# Load Tokenizer & Model
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    load_in_8bit=True,
    device_map="auto"
)

# ---------------------------
# LoRA Configuration
# ---------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ---------------------------
# Load Dataset
# ---------------------------
dataset = load_dataset(
    "json",
    data_files={
        "train": TRAIN_FILE,
        "validation": VAL_FILE
    }
)

# ---------------------------
# Tokenization
# ---------------------------
def format_instruction(example):
    text = (
        "<bos><start_of_turn>user\n"
        f"{example['instruction']}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
        f"{example['response']}\n"
        "<end_of_turn>"
    )
    return {"text": text}

dataset = dataset.map(format_instruction)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

tokenized_ds = dataset.map(
    tokenize,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# ---------------------------
# Data Collator
# ---------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# ---------------------------
# Training Arguments
# ---------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="steps",
    save_strategy="steps",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    fp16=True,
    logging_steps=50,
    eval_steps=100,
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none"
)

# ---------------------------
# Trainer
# ---------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)]
)

# ---------------------------
# Start Training
# ---------------------------
print("🚀 Starting Gemma-300M instruction fine-tuning...")
train_result = trainer.train()

print("✅ Training completed")
print(train_result.metrics)

# ---------------------------
# Save Model & Adapter
# ---------------------------
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"📦 Model saved to {OUTPUT_DIR}")
