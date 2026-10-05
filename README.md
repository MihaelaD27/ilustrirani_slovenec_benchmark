# Multimodal Understanding of Historical Slovenian Newspapers

This repository contains the benchmark data and experimental code for multimodal understanding of historical Slovenian newspaper pages from *Ilustrirani Slovenec*.

The benchmark covers six multimodal document-understanding tasks involving image-region localization, captioning, page-level text extraction, and full-page understanding.

The repository contains:

- predefined training and test JSONL splits for all six benchmark tasks,
- scripts for zero-shot and one-shot inference,
- supervised fine-tuning (SFT) scripts,
- SFT with task oversampling,
- LoRA merging,
- inference with fine-tuned models,
- task-specific evaluation code,
- and a script for downloading and converting the original newspaper issues into page images.

---

## Dataset

The benchmark contains six tasks:

1. **Dataset 1 — Image region → caption**  
   Given an image-region bounding box, generate the corresponding caption.

2. **Dataset 2 — Caption → image region**  
   Given an image caption, predict the corresponding bounding box.

3. **Dataset 3 — Image-region detection**  
   Detect all image regions on the newspaper page.

4. **Dataset 4 — Image-region detection and captioning**  
   Detect all image regions and generate the corresponding captions.

5. **Dataset 5 — Page-level text extraction**  
   Extract page text while excluding image captions.

6. **Dataset 6 — Full-page understanding**  
   Extract page text together with image-region locations and corresponding captions.

The benchmark uses a predefined page-level split of:

```text
2,300 training pages
256 test pages
```

The same page-level partition is used consistently across the six tasks so that no newspaper page occurs in both the training and test partitions.

Datasets 1 and 2 contain individual image-caption examples, while Datasets 3–6 operate at page level.

The dataset files are stored in:

```text
datasets/
```

with the following structure:

```text
dataset1_train.jsonl
dataset1_test.jsonl
dataset2_train.jsonl
dataset2_test.jsonl
dataset3_train.jsonl
dataset3_test.jsonl
dataset4_train.jsonl
dataset4_test.jsonl
dataset5_train.jsonl
dataset5_test.jsonl
dataset6_train.jsonl
dataset6_test.jsonl
```

The `*_train.jsonl` files form the benchmark training split. They are used for supervised fine-tuning and, in the one-shot setting, for selecting demonstrations.

The `*_test.jsonl` files are used for model evaluation.

Bounding boxes use normalized coordinates:

```text
[x1, y1, x2, y2]
```

with coordinate values in the range `[0, 1]`.

---

## Obtaining the newspaper images

The original newspaper PDFs and page images are not redistributed in this repository.

The corresponding *Ilustrirani Slovenec* issues are publicly available through the Digital Library of Slovenia (dLib.si):

https://www.dlib.si/details/URN:NBN:SI:spr-XWQWZFUW

To download the newspaper issues and convert them into individual page images, run:

```bash
python download_and_split_ilustrirani_slovenec.py
```

The script:

- finds the corresponding *Ilustrirani Slovenec* issues on dLib.si,
- downloads the PDF files,
- splits the PDFs into individual pages,
- renders each page as a 300-DPI PNG image,
- and generates filenames corresponding to the image references used in the benchmark.

By default, it creates:

```text
downloaded_pdfs/
images/
```

---

## Models

The experiments use the following vision-language model families:

- Gemma 3 4B
- Gemma 3 12B
- Gemma 3 27B
- SVILA 1 4B
- SVILA 1 12B
- EuroVLM 9B

The corresponding public model checkpoints can be obtained from Hugging Face.

For example:

```text
google/gemma-3-4b-it
google/gemma-3-12b-it
google/gemma-3-27b-it

GaMS-Beta/SVILA-1-4B
GaMS-Beta/SVILA-1-12B
```

For EuroVLM, set `MODEL_ID` or `model_id` to the corresponding public `EuroVLM-9B-Preview` checkpoint or local model path.

Model checkpoints are configured at the top of the individual scripts.

---

## Requirements

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

The main dependencies are:

```text
torch
transformers
datasets
peft
trl
bitsandbytes
vllm
pandas
numpy
Pillow
jiwer
PyMuPDF
requests
beautifulsoup4
```

For exact reproducibility, package versions should match the environment used for the experiments.

---

## Running the scripts

GPU selection can be controlled using `CUDA_VISIBLE_DEVICES`.

For example, using one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_sft.py
```

Using two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_zero_shot.py
```

Using four GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/run_one_shot.py
```

For vLLM inference, the number of selected GPUs should match the value of:

```python
TENSOR_PARALLEL_SIZE
```

in the corresponding inference script.

For example:

```python
TENSOR_PARALLEL_SIZE = 2
```

can be run with:

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_zero_shot.py
```

Model paths, output directories, batch size, tensor-parallel size, and optional dataset selection can be configured at the top of the scripts.

---

## Zero-shot inference

For Gemma 3 and SVILA models:

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_zero_shot.py
```

The model receives only:

```text
target image
+ task prompt
```

No demonstrations from the training split are provided.

Generation is deterministic:

```text
temperature = 0
top_p = 1
```

Set the desired checkpoint at the top of the script, for example:

```python
model_id = "google/gemma-3-12b-it"
```

or:

```python
model_id = "GaMS-Beta/SVILA-1-12B"
```

---

## One-shot inference

For Gemma 3 and SVILA models:

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_one_shot.py
```

For each test example, one example from the corresponding training task is sampled as a multimodal demonstration.

The demonstration contains:

```text
demonstration image
+ demonstration question
+ gold answer
```

followed by:

```text
target image
+ target question
```

Demonstration sampling uses:

```text
random seed = 42
```

for reproducibility.

---

## EuroVLM zero-shot and one-shot inference

EuroVLM uses a separate inference script because of its model-specific chat format and generation tokens.

Run:

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_zero_one_shot_eurovlm.py
```

The script supports:

```text
zero-shot
one-shot
```

inference.

The desired mode and model checkpoint can be configured at the top of the script.

---

## Standard supervised fine-tuning

For Gemma 3 and SVILA:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_sft.py
```

The six benchmark training splits are concatenated once:

```text
Dataset 1 x1
Dataset 2 x1
Dataset 3 x1
Dataset 4 x1
Dataset 5 x1
Dataset 6 x1
```

No task oversampling is applied in this script.

The script uses QLoRA with the following main configuration:

```text
4-bit quantization: NF4
double quantization: enabled
compute dtype: bfloat16

LoRA rank: 16
LoRA alpha: 32
LoRA dropout: 0.05

learning rate: 2e-4
epochs: 1
per-device batch size: 1
gradient accumulation: 16
optimizer: paged AdamW 8-bit
```

LoRA adapters are applied to:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

Training loss is computed over the complete formatted conversation, with padding tokens masked from the loss.

Configure the model at the top of the script, for example:

```python
MODEL_ID = "google/gemma-3-12b-it"
```

or:

```python
MODEL_ID = "GaMS-Beta/SVILA-1-12B"
```

---

## Supervised fine-tuning with task oversampling

Run:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_sft_oversampling.py
```

This script uses the same SFT configuration but changes the relative frequency of the six training tasks.

The oversampling factors are:

```text
Dataset 1: x1
Dataset 2: x1
Dataset 3: x3
Dataset 4: x4
Dataset 5: x4
Dataset 6: x4
```

The resulting combined training dataset is shuffled using:

```text
seed = 42
```

The source JSONL files are the same benchmark training splits; oversampling is performed inside the training script.

---

## EuroVLM supervised fine-tuning

For EuroVLM:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_eurovlm_sft.py
```

This script uses the same general QLoRA setup together with EuroVLM-specific model and processor handling.

It uses:

```text
LlavaNextForConditionalGeneration
```

and resizes images so that the maximum image side is:

```text
1280 pixels
```

before processing.

---

## Merging LoRA adapters

The SFT scripts save trained LoRA/PEFT adapters.

To merge an adapter into its corresponding base model before inference, run:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/merge_lora.py
```

Configure:

```python
BASE_MODEL_ID = "..."
ADAPTER_PATH = "..."
MERGED_OUT = "..."
```

For example:

```python
BASE_MODEL_ID = "google/gemma-3-12b-it"
ADAPTER_PATH = "gemma-3-12b-it-pdf-sft-full-loss-final"
MERGED_OUT = "gemma-3-12b-it-pdf-sft-full-loss-merged"
```

The script:

1. loads the base model in bfloat16,
2. loads the LoRA adapter,
3. calls `merge_and_unload()`,
4. saves the merged model,
5. saves the corresponding processor.

The script contains separate model-loading paths for Gemma 3 / SVILA and EuroVLM.

---

## Fine-tuned model inference

### Gemma 3 and SVILA

Run:

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_sft_inference.py
```

This script evaluates merged fine-tuned Gemma 3 and SVILA models.

At inference time, the model receives only the original benchmark task prompt and target image.

No demonstrations are added.

Configure the merged model at the top of the script:

```python
model_id = "gemma-3-12b-it-pdf-sft-full-loss-merged"
```

For an oversampled SFT model, for example:

```python
model_id = "gemma-3-12b-it-pdf-sft-full-loss-merged_oversampling"
```

### EuroVLM

Run:

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_sft_inference_eurovlm.py
```

EuroVLM is evaluated separately because it uses model-specific generation and stop tokens.

---

## Evaluation

Model predictions are evaluated using:

```bash
python scripts/evaluate.py
```

Add one or more directories containing generated `.tsv` prediction files to:

```python
RESULT_DIRS = [
    "results/..."
]
```

The script automatically identifies the task from the result filename and applies the corresponding evaluation procedure.

### Dataset 1 — Image region → caption

Metrics:

```text
CER
WER
```

### Dataset 2 — Caption → image region

Metric:

```text
Average IoU
```

### Dataset 3 — Image-region detection

Metrics:

```text
Average matched IoU
Precision@0.5
Recall@0.5
F1@0.5
```

### Dataset 4 — Image-region detection and captioning

Metrics:

```text
Average matched IoU
Precision@0.5
Recall@0.5
F1@0.5
Caption CER
Caption WER
```

Caption quality is evaluated using captions associated with matched image regions.

### Dataset 5 — Page-level text extraction

Metrics:

```text
CER
WER
```

### Dataset 6 — Full-page understanding

Metrics:

```text
Page CER
Page WER
Average matched IoU
Precision@0.5
Recall@0.5
F1@0.5
Caption CER
Caption WER
```

The evaluation script also records diagnostic information such as:

```text
invalid bounding-box predictions
non-normalized bounding boxes
missing page-text sections
hallucinated page-text sections
```

Per-example evaluation CSV files are generated together with the aggregate evaluation results.

---

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── download_and_split_ilustrirani_slovenec.py
│
├── datasets/
│   ├── dataset1_train.jsonl
│   ├── dataset1_test.jsonl
│   ├── dataset2_train.jsonl
│   ├── dataset2_test.jsonl
│   ├── dataset3_train.jsonl
│   ├── dataset3_test.jsonl
│   ├── dataset4_train.jsonl
│   ├── dataset4_test.jsonl
│   ├── dataset5_train.jsonl
│   ├── dataset5_test.jsonl
│   ├── dataset6_train.jsonl
│   └── dataset6_test.jsonl
│
└── scripts/
    ├── run_zero_shot.py
    ├── run_one_shot.py
    ├── run_zero_one_shot_eurovlm.py
    ├── train_sft.py
    ├── train_sft_oversampling.py
    ├── train_eurovlm_sft.py
    ├── merge_lora.py
    ├── run_sft_inference.py
    ├── run_sft_inference_eurovlm.py
    └── evaluate.py
```

---

## Source material

The benchmark is based on *Ilustrirani Slovenec*, a historical illustrated weekly supplement of the newspaper *Slovenec* published between 1924 and 1932.

The source newspaper issues are available through the Digital Library of Slovenia:

https://www.dlib.si/details/URN:NBN:SI:spr-XWQWZFUW

The original PDFs and page images are not redistributed in this repository.

---

## Citation

Citation information will be added after publication.
