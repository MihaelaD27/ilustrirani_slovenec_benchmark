import os
import torch

from transformers import (
    AutoProcessor,
    Gemma3ForConditionalGeneration,
    LlavaNextForConditionalGeneration,
)
from peft import PeftModel


# ============================================================
# CONFIG
# ============================================================

# Supported model types:
# "gemma"   -> Gemma 3 and SVILA
# "eurovlm" -> EuroVLM
MODEL_TYPE = "gemma"

# Example base model checkpoints:
# "gemma-3-4b-it"
# "gemma-3-12b-it"
# "gemma-3-27b-it"
# "SVILA-1-4B"
# "SVILA-1-12B"
# "EuroVLM-9B-Preview"
BASE_MODEL_ID = "gemma-3-4b-it"

# Path to the trained LoRA adapter.
#
# Standard SFT example:
# "gemma-3-4b-it-pdf-sft-full-loss-final"
#
# Oversampling example:
# "gemma-3-4b-it-pdf-sft-full-loss-final_oversampling"
#
# EuroVLM example:
# "eurovlm-pdf-sft-full-loss-final"
ADAPTER_PATH = "gemma-3-4b-it-pdf-sft-full-loss-final"

# Output path for the merged model.
MERGED_OUT = "gemma-3-4b-it-pdf-sft-full-loss-merged"


# ============================================================
# VALIDATE CONFIG
# ============================================================

if MODEL_TYPE not in {"gemma", "eurovlm"}:
    raise ValueError(
        "MODEL_TYPE must be either 'gemma' or 'eurovlm'."
    )


# ============================================================
# LOAD PROCESSOR
# ============================================================

print("Loading processor...")

if MODEL_TYPE == "eurovlm":
    processor = AutoProcessor.from_pretrained(
        ADAPTER_PATH,
        trust_remote_code=True,
    )
else:
    processor = AutoProcessor.from_pretrained(
        ADAPTER_PATH
    )


# ============================================================
# LOAD BASE MODEL
# ============================================================

print(f"Loading base model: {BASE_MODEL_ID}")

if MODEL_TYPE == "eurovlm":
    base_model = (
        LlavaNextForConditionalGeneration
        .from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    )

else:
    base_model = (
        Gemma3ForConditionalGeneration
        .from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    )


# ============================================================
# LOAD LORA ADAPTER
# ============================================================

print(f"Loading LoRA adapter: {ADAPTER_PATH}")

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
)


# ============================================================
# MERGE
# ============================================================

print("Merging LoRA adapter into base model...")

merged_model = model.merge_and_unload()


# ============================================================
# SAVE MERGED MODEL
# ============================================================

print(f"Saving merged model to: {MERGED_OUT}")

merged_model.save_pretrained(
    MERGED_OUT,
    safe_serialization=True,
    max_shard_size="5GB",
)

processor.save_pretrained(
    MERGED_OUT
)

print("✓ Merge complete.")