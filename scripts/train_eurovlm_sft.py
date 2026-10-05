import os
import torch
from datasets import load_dataset, concatenate_datasets
from PIL import Image

from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from transformers import (
    AutoProcessor,
    LlavaNextForConditionalGeneration,
    BitsAndBytesConfig,
)

# ============================================================
# CONFIG
# ============================================================

model_id = "utter-project/EuroVLM-9B-Preview"

DATA_DIR = "datasets"

TRAIN_FILES = [
    os.path.join(DATA_DIR, "dataset1_bb_2_caption_train.jsonl"),
    os.path.join(DATA_DIR, "dataset2_caption_2_bb_train.jsonl"),
    os.path.join(DATA_DIR, "dataset3_all_bb_train.jsonl"),
    os.path.join(DATA_DIR, "dataset4_all_bb_and_captions2_train.jsonl"),
    os.path.join(DATA_DIR, "dataset5_metatext_train.jsonl"),
    os.path.join(DATA_DIR, "dataset6_retrieve_all2_train.jsonl"),
]

IMAGE_ROOT = DATA_DIR
MAX_IMAGE_SIDE = 1280

OUTPUT_DIR = "./eurovlm-pdf-sft-full-loss-final"
TRAIN_OUTPUT_DIR = "./eurovlm-pdf-sft-full-loss-output"

# For testing the script before full training.
# Set to None for full training.
DEBUG_ONLY_FIRST_N = None


# ============================================================
# LOAD DATASETS
# ============================================================

print("Loading training datasets...")

datasets = []

for file_path in TRAIN_FILES:
    print(f"Loading: {file_path}")

    ds = load_dataset(
        "json",
        data_files=file_path,
        split="train",
    )

    if DEBUG_ONLY_FIRST_N is not None:
        ds = ds.select(
            range(
                min(
                    DEBUG_ONLY_FIRST_N,
                    len(ds),
                )
            )
        )

    datasets.append(ds)

# Standard SFT:
# each task dataset is included exactly once.
# No oversampling is applied.
train_dataset = concatenate_datasets(
    datasets
)

print(train_dataset)
print("Example:")
print(train_dataset[0])


# ============================================================
# LOAD PROCESSOR
# ============================================================

print("Loading processor...")

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

print(
    f"Processor class: "
    f"{processor.__class__.__name__}"
)

if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = (
        processor.tokenizer.eos_token
    )
    processor.tokenizer.pad_token_id = (
        processor.tokenizer.eos_token_id
    )


# ============================================================
# HELPERS
# ============================================================

def load_and_resize(
    path,
    max_side=MAX_IMAGE_SIDE,
):
    image = Image.open(
        path
    ).convert("RGB")

    if max(image.size) > max_side:
        ratio = (
            max_side / max(image.size)
        )

        image = image.resize(
            (
                int(image.width * ratio),
                int(image.height * ratio),
            )
        )

    return image


def strip_image_token(text):
    return (
        str(text)
        .replace("<image>\n", "")
        .replace("<image>", "")
        .strip()
    )


def extract_prompt_and_truth(example):
    conversation = example["conversations"]

    if (
        isinstance(conversation, list)
        and len(conversation) > 0
        and isinstance(conversation[0], list)
    ):
        conversation = conversation[0]

    prompt = strip_image_token(
        conversation[0]["value"]
    )

    truth = strip_image_token(
        conversation[1]["value"]
    )

    return prompt, truth


def format_with_chat_template(example):
    """
    Uses only the prompt already present in the dataset.

    Loss is computed over the whole formatted conversation:
    user prompt + assistant answer.
    """

    prompt, truth = extract_prompt_and_truth(
        example
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": truth,
                },
            ],
        },
    ]

    formatted_text = (
        processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    )

    return {
        "formatted_text": formatted_text,
    }


print("Applying chat template...")

train_dataset = train_dataset.map(
    format_with_chat_template
)


# ============================================================
# MULTIMODAL COLLATOR
# ============================================================

def collate_fn(examples):
    texts = [
        ex["formatted_text"]
        for ex in examples
    ]

    images = []

    for ex in examples:
        image_path = ex["image_path"]

        if not os.path.isabs(image_path):
            image_path = os.path.join(
                IMAGE_ROOT,
                image_path,
            )

        image = load_and_resize(
            image_path
        )

        images.append(image)

    batch = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt",
    )

    # Compute loss over the entire formatted conversation.
    labels = batch["input_ids"].clone()

    # Ignore padding tokens in the loss.
    pad_token_id = (
        processor.tokenizer.pad_token_id
    )

    if pad_token_id is not None:
        labels[
            labels == pad_token_id
        ] = -100

    batch["labels"] = labels

    return batch


# ============================================================
# QLORA 4-BIT CONFIG
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading EuroVLM-9B in 4-bit...")

model = (
    LlavaNextForConditionalGeneration
    .from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
    )
)

model.config.pad_token_id = (
    processor.tokenizer.pad_token_id
)

print(model.hf_device_map)

# EuroVLM / LLaVA-NeXT processor configuration.
processor.image_processor.patch_size = (
    model.config.vision_config.patch_size
)

processor.image_processor.vision_feature_select_strategy = (
    model.config.vision_feature_select_strategy
)

processor.num_additional_image_tokens = 0

print(
    f"Model class: "
    f"{model.__class__.__name__}"
)

print(
    f"image_token_index: "
    f"{getattr(model.config, 'image_token_index', None)}"
)

print(
    f"processor image_token: "
    f"{getattr(processor, 'image_token', None)}"
)


# ============================================================
# LORA CONFIG
# ============================================================

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
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


# ============================================================
# TRAINING CONFIG
# ============================================================

training_args = SFTConfig(
    output_dir=TRAIN_OUTPUT_DIR,

    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,

    learning_rate=2e-4,
    num_train_epochs=1,

    logging_steps=10,
    save_strategy="epoch",

    optim="paged_adamw_8bit",
    bf16=True,

    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={
        "use_reentrant": False,
    },

    remove_unused_columns=False,
    dataset_kwargs={
        "skip_prepare_dataset": True,
    },

    report_to="none",
)


# ============================================================
# TRAINER
# ============================================================

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    peft_config=peft_config,
    args=training_args,
    data_collator=collate_fn,
    processing_class=processor.tokenizer,
)

trainer.model.print_trainable_parameters()


# ============================================================
# TRAIN
# ============================================================

print("Starting SFT training...")

trainer.train()


# ============================================================
# SAVE ADAPTER + PROCESSOR
# ============================================================

print(
    f"Saving model to: "
    f"{OUTPUT_DIR}"
)

trainer.save_model(
    OUTPUT_DIR
)

processor.save_pretrained(
    OUTPUT_DIR
)

print("Training complete.")
