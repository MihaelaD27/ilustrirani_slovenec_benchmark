import os
import json
import pandas as pd
import warnings
from PIL import Image

from vllm import LLM, SamplingParams

warnings.filterwarnings("ignore")

# =========================
# CONFIG
# =========================

# Example merged SFT checkpoints:
# "gemma-3-4b-it-pdf-sft-full-loss-merged"
# "gemma-3-12b-it-pdf-sft-full-loss-merged"
# "gemma-3-27b-it-pdf-sft-full-loss-merged"
# "SVILA-1-4B-pdf-sft-full-loss-merged"
# "SVILA-1-12B-pdf-sft-full-loss-merged"
#
# Oversampling examples:
# "gemma-3-4b-it-pdf-sft-full-loss-merged_oversampling"
# "gemma-3-12b-it-pdf-sft-full-loss-merged_oversampling"
# "gemma-3-27b-it-pdf-sft-full-loss-merged_oversampling"

model_id = "gemma-3-4b-it-pdf-sft-full-loss-merged"

IMAGE_ROOT = "datasets"

model_name = os.path.basename(model_id.rstrip("/"))

OUTPUT_DIR = f"results/{model_name}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Adjust depending on model size and available GPUs.
TENSOR_PARALLEL_SIZE = 2

DTYPE = "bfloat16"
BATCH_SIZE = 4

DEBUG_ONLY_FIRST_N = None

# Set to None to run all datasets.
# Example: ["dataset6_retrieve_all2"]
RUN_ONLY_DATASETS = None


# =========================
# DATASETS
# =========================

DATASETS = [
    {
        "name": "dataset1_bb_2_caption",
        "test": "datasets/dataset1_bb_2_caption_test.jsonl",
    },
    {
        "name": "dataset2_caption_2_bb",
        "test": "datasets/dataset2_caption_2_bb_test.jsonl",
    },
    {
        "name": "dataset3_all_bb",
        "test": "datasets/dataset3_all_bb_test.jsonl",
    },
    {
        "name": "dataset4_all_bb_and_captions2",
        "test": "datasets/dataset4_all_bb_and_captions2_test.jsonl",
    },
    {
        "name": "dataset5_metatext",
        "test": "datasets/dataset5_metatext_test.jsonl",
    },
    {
        "name": "dataset6_retrieve_all2",
        "test": "datasets/dataset6_retrieve_all2_test.jsonl",
    },
]


# =========================
# HELPERS
# =========================

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def normalize_ws(text):
    return " ".join(str(text).split())


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

    real_prompt = strip_image_token(
        conversation[0]["value"]
    )

    truth = strip_image_token(
        conversation[1]["value"]
    )

    return real_prompt, truth


def clean_generation(text):
    text = str(text)

    for marker in [
        "<end_of_turn>",
        "<eos>",
        "<start_of_turn>user",
        "<start_of_turn>model",
        "<start_of_turn>assistant",
    ]:
        if marker in text:
            text = text.split(marker)[0]

    text = text.replace("<bos>", "").strip()

    if text.startswith("Odgovor:"):
        text = text[len("Odgovor:"):].strip()

    return text.strip()


def get_max_new_tokens(dataset_name):
    return {
        "dataset1_bb_2_caption": 800,
        "dataset2_caption_2_bb": 256,
        "dataset3_all_bb": 1000,
        "dataset4_all_bb_and_captions2": 3000,
        "dataset5_metatext": 3000,
        "dataset6_retrieve_all2": 3000,
    }.get(dataset_name, 512)


def get_repetition_penalty(dataset_name):
    return {
        "dataset1_bb_2_caption": 1.1,
        "dataset2_caption_2_bb": 1.1,
        "dataset3_all_bb": 1.05,
        "dataset4_all_bb_and_captions2": 1.1,
        "dataset5_metatext": 1.1,
        "dataset6_retrieve_all2": 1.15,
    }.get(dataset_name, 1.1)


# =========================
# LOAD MODEL
# =========================

print("Loading merged SFT model with vLLM...")
print(f"Model: {model_id}")
print(f"Tensor parallel size: {TENSOR_PARALLEL_SIZE}")

llm = LLM(
    model=model_id,
    dtype=DTYPE,
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    gpu_memory_utilization=0.7,
    trust_remote_code=True,
    max_model_len=16384,
    limit_mm_per_prompt={"image": 1},
    skip_mm_profiling=True,
    max_num_seqs=BATCH_SIZE,
    enforce_eager=True,
    disable_log_stats=True,
    disable_custom_all_reduce=True,
)

print("✓ Model loaded successfully")


# =========================
# MAIN LOOP
# =========================

for dataset_config in DATASETS:
    dataset_name = dataset_config["name"]

    if (
        RUN_ONLY_DATASETS
        and dataset_name not in RUN_ONLY_DATASETS
    ):
        continue

    print(f"\nRunning dataset: {dataset_name}")

    test_data = load_jsonl(
        dataset_config["test"]
    )

    test_iter = (
        test_data[:DEBUG_ONLY_FIRST_N]
        if DEBUG_ONLY_FIRST_N
        else test_data
    )

    results = []
    skipped = 0

    max_new_tokens = get_max_new_tokens(
        dataset_name
    )

    repetition_penalty = get_repetition_penalty(
        dataset_name
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        stop=[
            "<end_of_turn>",
            "<eos>",
        ],
    )

    for batch_start in range(
        0,
        len(test_iter),
        BATCH_SIZE,
    ):
        batch = test_iter[
            batch_start:
            batch_start + BATCH_SIZE
        ]

        batch_requests = []
        batch_examples = []

        for local_idx, example in enumerate(
            batch
        ):
            global_idx = (
                batch_start
                + local_idx
                + 1
            )

            example_id = example.get(
                "id",
                f"example_{global_idx}",
            )

            image_path = example["image_path"]

            full_image_path = image_path

            if not os.path.isabs(
                full_image_path
            ):
                full_image_path = os.path.join(
                    IMAGE_ROOT,
                    image_path,
                )

            if not os.path.exists(
                full_image_path
            ):
                print(
                    f"Missing image: "
                    f"{full_image_path}"
                )
                skipped += 1
                continue

            real_prompt, truth = (
                extract_prompt_and_truth(
                    example
                )
            )

            # SFT inference uses only the original
            # benchmark prompt from the dataset.
            full_prompt = real_prompt

            image_pil = Image.open(
                full_image_path
            ).convert("RGB")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_pil",
                            "image_pil": image_pil,
                        },
                        {
                            "type": "text",
                            "text": full_prompt,
                        },
                    ],
                }
            ]

            batch_requests.append(
                messages
            )

            batch_examples.append(
                {
                    "idx": global_idx,
                    "id": example_id,
                    "image_path": image_path,
                    "prompt": full_prompt,
                    "truth": truth,
                }
            )

        if not batch_requests:
            continue

        print(
            f"\nBatch "
            f"{batch_start + 1}-"
            f"{batch_start + len(batch_examples)} "
            f"/ {len(test_iter)} | "
            f"batch size = {len(batch_examples)}"
        )

        outputs = llm.chat(
            batch_requests,
            sampling_params=sampling_params,
        )

        for out_idx, example_info in enumerate(
            batch_examples
        ):
            decoded = clean_generation(
                outputs[
                    out_idx
                ].outputs[0].text
            )

            prompt_clean = normalize_ws(
                example_info["prompt"]
            )

            truth_clean = normalize_ws(
                example_info["truth"]
            )

            answer_clean = normalize_ws(
                decoded
            )

            print(
                f"\n"
                f"[{example_info['idx']}/"
                f"{len(test_iter)}] "
                f"{example_info['id']}"
            )

            print(
                f"Image:  "
                f"{example_info['image_path']}"
            )

            print(
                f"Truth:  "
                f"{truth_clean[:180]}..."
            )

            print(
                f"Answer: "
                f"{answer_clean[:180]}..."
            )

            results.append(
                (
                    example_info["id"],
                    example_info["image_path"],
                    prompt_clean,
                    truth_clean,
                    answer_clean,
                )
            )

    out_path = os.path.join(
        OUTPUT_DIR,
        f"{dataset_name}_sft.tsv",
    )

    df = pd.DataFrame(
        results,
        columns=[
            "id",
            "image_path",
            "prompt",
            "true",
            "answer",
        ],
    )

    df.to_csv(
        out_path,
        sep="\t",
        index=False,
        encoding="utf-8",
    )

    print(f"\n✓ Saved: {out_path}")

    print(
        f"Summary: "
        f"{len(results)} done, "
        f"{skipped} skipped"
    )

print("\n✓ All SFT vLLM testing complete.")