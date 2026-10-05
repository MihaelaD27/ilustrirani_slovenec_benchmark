import os
import json
import random
import pandas as pd
import warnings
from PIL import Image
from vllm import LLM, SamplingParams

warnings.filterwarnings("ignore")

# =========================
# CONFIG
# =========================

# EuroVLM checkpoint
model_id = "EuroVLM-9B-Preview"

IMAGE_ROOT = "datasets"

# Adjust depending on available GPUs.
TENSOR_PARALLEL_SIZE = 4

DTYPE = "bfloat16"
BATCH_SIZE = 2

# Run zero-shot, one-shot, or both.
SHOT_MODES = ["zero", "one"]

RANDOM_SEED = 42
DEBUG_ONLY_FIRST_N = None

# Set to None to run all datasets.
# Example: ["dataset6_retrieve_all2"]
RUN_ONLY_DATASETS = None

random.seed(RANDOM_SEED)

# =========================
# DATASETS
# =========================

DATASETS = [
    {
        "name": "dataset1_bb_2_caption",
        "train": "datasets/dataset1_bb_2_caption_train.jsonl",
        "test": "datasets/dataset1_bb_2_caption_test.jsonl",
    },
    {
        "name": "dataset2_caption_2_bb",
        "train": "datasets/dataset2_caption_2_bb_train.jsonl",
        "test": "datasets/dataset2_caption_2_bb_test.jsonl",
    },
    {
        "name": "dataset3_all_bb",
        "train": "datasets/dataset3_all_bb_train.jsonl",
        "test": "datasets/dataset3_all_bb_test.jsonl",
    },
    {
        "name": "dataset4_all_bb_and_captions2",
        "train": "datasets/dataset4_all_bb_and_captions2_train.jsonl",
        "test": "datasets/dataset4_all_bb_and_captions2_test.jsonl",
    },
    {
        "name": "dataset5_metatext",
        "train": "datasets/dataset5_metatext_train.jsonl",
        "test": "datasets/dataset5_metatext_test.jsonl",
    },
    {
        "name": "dataset6_retrieve_all2",
        "train": "datasets/dataset6_retrieve_all2_train.jsonl",
        "test": "datasets/dataset6_retrieve_all2_test.jsonl",
    },
]

# =========================
# LOAD MODEL
# =========================

print("Loading EuroVLM with vLLM...")
print(f"Model: {model_id}")
print(f"Tensor parallel size: {TENSOR_PARALLEL_SIZE}")

llm = LLM(
    model=model_id,
    dtype=DTYPE,
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    gpu_memory_utilization=0.9,
    trust_remote_code=True,
    max_model_len=16384,

    # Zero-shot uses 1 image.
    # One-shot uses 1 demonstration image + 1 target image.
    limit_mm_per_prompt={"image": 2},

    enforce_eager=True,
    disable_log_stats=True,
    disable_custom_all_reduce=True,
)

print("✓ Model loaded successfully")

# =========================
# HELPERS
# =========================

def clean_generation(text):
    text = str(text)

    for marker in [
        "<|im_end|>",
        "<|im_start|>user",
        "<|im_start|>assistant",
    ]:
        if marker in text:
            text = text.split(marker)[0]

    text = (
        text
        .replace("<bos>", "")
        .replace("<s>", "")
        .strip()
    )

    if text.startswith("Odgovor:"):
        text = text[len("Odgovor:"):].strip()

    return text.strip()


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


def get_num_shots(shot_mode):
    return {
        "zero": 0,
        "one": 1,
    }[shot_mode]


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


def sample_few_shots(
    train_data,
    num_shots,
    exclude_id=None,
):
    if num_shots == 0:
        return []

    filtered = [
        ex for ex in train_data
        if ex.get("id") != exclude_id
    ]

    if not filtered:
        return []

    return random.sample(
        filtered,
        min(num_shots, len(filtered)),
    )


def build_fewshot_block(examples):
    if not examples:
        return ""

    text = "\n\nPrimeri:\n"

    for i, ex in enumerate(examples, 1):
        ex_q, ex_a = extract_prompt_and_truth(ex)

        text += (
            f"\nPrimer {i}:\n"
            f"Vprašanje: {ex_q}\n"
            f"Odgovor: {ex_a}\n"
        )

    return text


def build_full_prompt(
    dataset_name,
    real_prompt,
    fewshot_examples=None,
):
    common_prefix = """Ti si API za obdelavo PDF dokumentov.
Vse informacije moraš pridobiti IZKLJUČNO iz priložene slike.
Sledi natančno vprašanju iz dataseta.
NE ugibaj.
NE opisuj vizualne vsebine slike s svojimi besedami.
Uporabi samo podatke, ki so dejansko vidni na sliki.
Odgovori SAMO na vprašanje SAMO z rezultatom, ne dodajaj dodatnih informacij ali pojasnil.
Če je naloga povezana s koordinatami, so te v formatu [x1, y1, x2, y2] (normalizirano med 0 in 1)."""

    if dataset_name == "dataset1_bb_2_caption":
        prefix = """Ti si API za obdelavo PDF dokumentov.
Sledi natančno vprašanju iz dataseta.
Vse informacije moraš pridobiti IZKLJUČNO iz priložene slike.

Vrni SAMO caption/besedilo, ki pripada označeni sliki na podanih koordinatah.

Besedilo mora biti dejansko napisano na strani.
NE izmišljaj ali dopolnjuj besedila.
NE dodajaj besedila, ki ni vidno na sliki.
NE opisuj slike s svojimi besedami.
NE dodajaj razlage, opozoril ali dodatnega besedila.

Odgovori SAMO z rezultatom."""

    elif dataset_name == "dataset2_caption_2_bb":
        prefix = """Ti si API za obdelavo PDF dokumentov.

Vse informacije pridobi IZKLJUČNO iz priložene slike.

NE ugibaj.
NE dodajaj zunanjega znanja.
NE opisuj slike.
NE vračaj razlage.
NE dodajaj dodatnega besedila.

Vrni SAMO EN box v formatu:
[x1, y1, x2, y2]

Koordinate morajo biti normalizirane med 0 in 1.

Odgovori na vprašanje SAMO z rezultatom."""

    elif dataset_name == "dataset3_all_bb":
        prefix = """Ti si API za obdelavo PDF dokumentov.
Sledi natančno vprašanju iz dataseta.
Vse informacije moraš pridobiti IZKLJUČNO iz priložene slike.

Vrni SAMO seznam bounding box koordinat za VSE slike na strani.

Vsak bounding box mora biti v obliki:
[x1, y1, x2, y2]

Celoten odgovor mora biti Python 1 seznam, vse koordinate slik damo skupaj v en seznam na koncu!:
[[x1, y1, x2, y2],
 [x1, y1, x2, y2]]

Če je samo ena slika, vrni seznam z enim bounding boxom.

Koordinate morajo biti normalizirane med 0 in 1.
NE uporabljaj pixel koordinat.
NE dodajaj opisov, razlage, markdowna ali dodatnega besedila.
Odgovori SAMO z rezultatom."""

    elif dataset_name == "dataset4_all_bb_and_captions2":
        prefix = """Ti si API za obdelavo PDF dokumentov.

Vse informacije pridobi IZKLJUČNO iz priložene slike.

NE ugibaj.
NE dodajaj zunanjega znanja.
NE opisuj slik.
NE vračaj razlage.
NE dodajaj dodatnega besedila.

Poišči VSE slike na strani.

Za vsako sliko vrni:
Slika X koordinate: [x1, y1, x2, y2]
Slika X opis: <caption ali besedilo ob sliki>

'Slika opis' pomeni caption ali besedilo ob sliki.
NE pomeni vizualnega opisa slike.

Koordinate morajo biti normalizirane med 0 in 1.
NIKOLI ne uporabljaj pixel koordinat.

Če slika nima captiona, napiši:
Slika X opis: Slika nima opisa.

Odgovori na vprašanje SAMO z rezultatom."""

    elif dataset_name == "dataset5_metatext":
        prefix = """Ti si API za obdelavo PDF dokumentov.
Sledi natančno vprašanju iz dataseta.
Vse informacije moraš pridobiti IZKLJUČNO iz priložene slike.

Vrni SAMO dobesedni prepis glavnega besedila strani, ki NI caption/opis slike.

NE vračaj captionov.
NE opisuj slik.
NE povzemaj.
NE dodajaj razlage, komentarjev, opozoril ali markdowna.

Če na strani ni glavnega besedila brez captionov, vrni TOČNO:
Stran nima besedila, ki ni povezano s slikami.

Odgovori SAMO z rezultatom."""

    elif dataset_name == "dataset6_retrieve_all2":
        prefix = """Ti si API za obdelavo PDF dokumentov.

Vse informacije pridobi IZKLJUČNO iz priložene slike.
NE ugibaj.
NE dodajaj zunanjega znanja.
NE opisuj slike.
NE povzemaj.
NE parafraziraj.

Vrni vsebino v istem formatu kot v učnih primerih.

FORMAT ODGOVORA MORA BITI TOČNO:
Če obstaja glavno besedilo strani, začni z:
Besedilo strani:
Če obstajajo slike, nadaljuj z:
Slike:
Slika 1 koordinate: [x1, y1, x2, y2]
Slika 1 opis: <caption ali besedilo ob sliki>

Koordinate morajo biti normalizirane med 0 in 1.

'Slika opis' pomeni caption ob sliki, NE vizualni opis.
'Besedilo strani' pomeni tekst, ki ni caption slike.

Če ni glavnega besedila, ne napiši 'Besedilo strani:'.
Če ni slik, ne napiši 'Slike:'.
Če slika nima captiona, napiši: Slika X opis: Slika nima opisa.

Odgovori na vprašanje SAMO z rezultatom."""

    else:
        prefix = common_prefix

    if not fewshot_examples:
        return f"""{prefix}

Vprašanje: {real_prompt}
Odgovor:"""

    fewshot_block = build_fewshot_block(
        fewshot_examples
    )

    return f"""{prefix}

{fewshot_block}

Zdaj odgovori na naslednje vprašanje.

Vprašanje: {real_prompt}
Odgovor:"""


def build_multimodal_content(
    full_prompt,
    fewshot_examples,
    real_prompt,
    image_pil,
):
    content = []

    # =========================
    # ONE-SHOT
    # =========================

    if fewshot_examples:
        prefix_part = full_prompt.split(
            "Primeri:"
        )[0]

        content.append({
            "type": "text",
            "text": (
                prefix_part
                + "\n\nPrimeri:\n"
            ),
        })

        for i, shot_ex in enumerate(
            fewshot_examples,
            1,
        ):
            shot_q, shot_a = (
                extract_prompt_and_truth(
                    shot_ex
                )
            )

            shot_image_path = (
                shot_ex["image_path"]
            )

            shot_full_image_path = os.path.join(
                IMAGE_ROOT,
                shot_image_path,
            )

            if not os.path.exists(
                shot_full_image_path
            ):
                raise FileNotFoundError(
                    f"Missing one-shot image: "
                    f"{shot_full_image_path}"
                )

            shot_image = Image.open(
                shot_full_image_path
            ).convert("RGB")

            content.append({
                "type": "text",
                "text": (
                    f"\nPrimer {i} slika:\n"
                ),
            })

            content.append({
                "type": "image_pil",
                "image_pil": shot_image,
            })

            content.append({
                "type": "text",
                "text": (
                    f"\nPrimer {i}:\n"
                    f"Vprašanje: {shot_q}\n"
                    f"Odgovor: {shot_a}\n"
                ),
            })

        content.append({
            "type": "text",
            "text": (
                "\nZdaj odgovori na "
                "naslednje vprašanje.\n"
            ),
        })

        content.append({
            "type": "image_pil",
            "image_pil": image_pil,
        })

        content.append({
            "type": "text",
            "text": (
                f"\nVprašanje: {real_prompt}\n"
                "Odgovor:"
            ),
        })

    # =========================
    # ZERO-SHOT
    # =========================

    else:
        content.append({
            "type": "image_pil",
            "image_pil": image_pil,
        })

        content.append({
            "type": "text",
            "text": full_prompt,
        })

    return content


# =========================
# MAIN LOOP
# =========================

for shot_mode in SHOT_MODES:
    if shot_mode == "zero":
        mode_name = "zero_shot"
    elif shot_mode == "one":
        mode_name = "one_shot"
    else:
        raise ValueError(
            f"Unsupported shot mode: {shot_mode}"
        )

    OUTPUT_DIR = (
        f"results/eurovlm_9b_{mode_name}"
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    print(
        f"\nRUNNING EUROVLM: "
        f"{mode_name.upper()}"
    )

    for dataset_config in DATASETS:
        dataset_name = (
            dataset_config["name"]
        )

        if (
            RUN_ONLY_DATASETS
            and dataset_name
            not in RUN_ONLY_DATASETS
        ):
            continue

        print(
            f"\nRunning dataset: "
            f"{dataset_name}"
        )

        test_data = load_jsonl(
            dataset_config["test"]
        )

        # Training data are needed only
        # for one-shot demonstrations.
        if shot_mode == "one":
            train_data = load_jsonl(
                dataset_config["train"]
            )
        else:
            train_data = []

        num_shots = get_num_shots(
            shot_mode
        )

        test_iter = (
            test_data[:DEBUG_ONLY_FIRST_N]
            if DEBUG_ONLY_FIRST_N
            else test_data
        )

        results = []
        skipped = 0

        # Max tokens per dataset
        max_tokens_per_dataset = {
            "dataset1_bb_2_caption": 800,
            "dataset2_caption_2_bb": 256,
            "dataset3_all_bb": 1000,
            "dataset4_all_bb_and_captions2": 3000,
            "dataset5_metatext": 3000,
            "dataset6_retrieve_all2": 3000,
        }

        max_new_tokens = (
            max_tokens_per_dataset.get(
                dataset_name,
                512,
            )
        )

        # Repetition penalty per dataset
        repetition_penalty_per_dataset = {
            "dataset1_bb_2_caption": 1.1,
            "dataset2_caption_2_bb": 1.1,
            "dataset3_all_bb": 1.05,
            "dataset4_all_bb_and_captions2": 1.1,
            "dataset5_metatext": 1.1,
            "dataset6_retrieve_all2": 1.15,
        }

        repetition_penalty = (
            repetition_penalty_per_dataset.get(
                dataset_name,
                1.1,
            )
        )

        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            stop=["<|im_end|>"],
        )

        # Process examples in batches
        for batch_start in range(
            0,
            len(test_iter),
            BATCH_SIZE,
        ):
            batch_end = min(
                batch_start + BATCH_SIZE,
                len(test_iter),
            )

            batch = test_iter[
                batch_start:batch_end
            ]

            batch_requests = []
            batch_examples = []

            for example in batch:
                example_id = example.get(
                    "id",
                    (
                        f"example_"
                        f"{batch_start + len(batch_examples) + 1}"
                    ),
                )

                image_path = (
                    example["image_path"]
                )

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

                fewshot_examples = (
                    sample_few_shots(
                        train_data,
                        num_shots,
                        exclude_id=example_id,
                    )
                )

                full_prompt = (
                    build_full_prompt(
                        dataset_name=dataset_name,
                        real_prompt=real_prompt,
                        fewshot_examples=(
                            fewshot_examples
                        ),
                    )
                )

                image_pil = Image.open(
                    full_image_path
                ).convert("RGB")

                content = (
                    build_multimodal_content(
                        full_prompt=full_prompt,
                        fewshot_examples=(
                            fewshot_examples
                        ),
                        real_prompt=real_prompt,
                        image_pil=image_pil,
                    )
                )

                messages = [{
                    "role": "user",
                    "content": content,
                }]

                batch_requests.append(
                    messages
                )

                batch_examples.append({
                    "id": example_id,
                    "image_path": image_path,
                    "full_prompt": full_prompt,
                    "truth": truth,
                })

            if not batch_requests:
                continue

            outputs = llm.chat(
                batch_requests,
                sampling_params=sampling_params,
            )

            for (
                out_idx,
                example_info,
            ) in enumerate(batch_examples):
                decoded = clean_generation(
                    outputs[
                        out_idx
                    ].outputs[0].text
                )

                prompt_clean = normalize_ws(
                    example_info[
                        "full_prompt"
                    ]
                )

                truth_clean = normalize_ws(
                    example_info["truth"]
                )

                answer_clean = normalize_ws(
                    decoded
                )

                print(
                    f"\n"
                    f"[{batch_start + out_idx + 1}] "
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

                results.append((
                    example_info["id"],
                    example_info[
                        "image_path"
                    ],
                    prompt_clean,
                    truth_clean,
                    answer_clean,
                ))

        out_path = os.path.join(
            OUTPUT_DIR,
            f"{dataset_name}_{mode_name}.tsv",
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

        print(
            f"\n✓ Saved: {out_path}"
        )

        print(
            f"Summary: "
            f"{len(results)} done, "
            f"{skipped} skipped"
        )

print(
    "\n✓ EuroVLM zero-shot and "
    "one-shot processing complete."
)