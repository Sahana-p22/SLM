# hf_training/train_lora.py

from datasets import load_dataset

from transformers import (

    AutoTokenizer,

    AutoModelForCausalLM,

    BitsAndBytesConfig,

    TrainingArguments
)

from peft import (

    LoraConfig,

    prepare_model_for_kbit_training
)

from trl import SFTTrainer

import torch

from model.hf_training.configs import (

    MODEL_PATH,

    DATASET_OUTPUT_PATH,

    LORA_OUTPUT_DIR
)


# =========================================================
# DEVICE INFO
# =========================================================

print(
    f"[INFO] CUDA Available: "
    f"{torch.cuda.is_available()}"
)

if torch.cuda.is_available():

    print(
        f"[INFO] GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )


# =========================================================
# LOAD TOKENIZER
# =========================================================

print(
    "\n[INFO] Loading Tokenizer..."
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"


# =========================================================
# QUANTIZATION CONFIG
# =========================================================

print(
    "\n[INFO] Configuring 4-bit Quantization..."
)

bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_compute_dtype=torch.float16,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_use_double_quant=True
)


# =========================================================
# LOAD BASE MODEL
# =========================================================

print(
    f"\n[INFO] Loading Base Model From {MODEL_PATH}..."
)

model = AutoModelForCausalLM.from_pretrained(

    MODEL_PATH,

    quantization_config=bnb_config,

    device_map="auto",

    trust_remote_code=True
)


# =========================================================
# PREPARE MODEL FOR QLORA
# =========================================================

print(
    "\n[INFO] Preparing Model For QLoRA..."
)

model = prepare_model_for_kbit_training(
    model
)


# =========================================================
# LORA CONFIG
# =========================================================

peft_config = LoraConfig(

    r=16,

    lora_alpha=32,

    target_modules=[

        "q_proj",

        "k_proj",

        "v_proj",

        "o_proj",

        "gate_proj",

        "up_proj",

        "down_proj"
    ],

    lora_dropout=0.05,

    bias="none",

    task_type="CAUSAL_LM"
)


# =========================================================
# LOAD DATASET
# =========================================================

print(
    "\n[INFO] Loading Dataset..."
)

dataset = load_dataset(

    "json",

    data_files=str(
        DATASET_OUTPUT_PATH
    )
)


# =========================================================
# TOKENIZATION
# =========================================================

MAX_LENGTH = 256


def tokenize_function(example):

    tokens = tokenizer(

        example["text"],

        truncation=True,

        padding="max_length",

        max_length=MAX_LENGTH
    )

    tokens["labels"] = tokens[
        "input_ids"
    ].copy()

    return tokens


print(
    "\n[INFO] Tokenizing Dataset..."
)

tokenized_dataset = dataset.map(

    tokenize_function,

    remove_columns=["text"]
)


# =========================================================
# TRAINING ARGS
# =========================================================

print(
    "\n[INFO] Configuring Training..."
)

training_args = TrainingArguments(

    output_dir=str(
        LORA_OUTPUT_DIR
    ),

    # -----------------------------------------------------
    # BATCHING
    # -----------------------------------------------------

    per_device_train_batch_size=1,

    gradient_accumulation_steps=2,

    # -----------------------------------------------------
    # TRAINING
    # -----------------------------------------------------

    learning_rate=2e-4,

    num_train_epochs=1,

    # -----------------------------------------------------
    # LOGGING
    # -----------------------------------------------------

    logging_steps=5,

    # -----------------------------------------------------
    # CHECKPOINTS
    # -----------------------------------------------------

    save_strategy="steps",

    save_steps=25,

    save_total_limit=2,

    # -----------------------------------------------------
    # PRECISION
    # -----------------------------------------------------

    fp16=False,

    bf16=False,

    # -----------------------------------------------------
    # STABILITY
    # -----------------------------------------------------

    max_grad_norm=0.3,

    # -----------------------------------------------------
    # OPTIMIZER
    # -----------------------------------------------------

    optim="paged_adamw_8bit",

    # -----------------------------------------------------
    # REPORTING
    # -----------------------------------------------------

    report_to="none",

    # -----------------------------------------------------
    # PERFORMANCE
    # -----------------------------------------------------

    torch_compile=False,

    # Windows has no fork(); multiprocess dataloader workers re-import and
    # re-execute this whole top-level script via spawn, so keep this at 0.
    dataloader_num_workers=0,

    # -----------------------------------------------------
    # STABILITY
    # -----------------------------------------------------

    remove_unused_columns=False
)


# =========================================================
# SFT TRAINER
# =========================================================

print(
    "\n[INFO] Initializing SFTTrainer..."
)

trainer = SFTTrainer(

    model=model,

    train_dataset=tokenized_dataset["train"],

    peft_config=peft_config,

    args=training_args
)


# =========================================================
# START TRAINING
# =========================================================

print(
    "\n[INFO] Starting QLoRA Training...\n"
)

trainer.train()


# =========================================================
# SAVE FINAL MODEL
# =========================================================

print(
    "\n[INFO] Saving Final LoRA Adapters..."
)

trainer.model.save_pretrained(
    LORA_OUTPUT_DIR
)

tokenizer.save_pretrained(
    LORA_OUTPUT_DIR
)

print(
    "\n[INFO] Assistant-Style QLoRA Training Complete"
)