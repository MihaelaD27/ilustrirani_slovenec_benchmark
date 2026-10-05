import os
import re
import ast
import csv
from collections import defaultdict

import numpy as np
import pandas as pd
from jiwer import cer, wer


# ============================================================
# CONFIG
# ============================================================

# Add one or more result directories to evaluate.
#
# Examples:
# "results/gemma-3-12b-it_zero_shot"
# "results/gemma-3-12b-it_one_shot"
# "results/gemma-3-12b-it-pdf-sft-full-loss-merged"
# "results/gemma-3-12b-it-pdf-sft-full-loss-merged_oversampling"
# "results/SVILA-1-12B_zero_shot"
# "results/eurovlm_9b_zero_shot"
# "results/eurovlm-pdf-sft-full-loss-merged"
RESULT_DIRS = [
    "results/gemma-3-12b-it_zero_shot",
]

OUTPUT_CSV = "evaluation_results.csv"


# ============================================================
# TEXT UTILITIES
# ============================================================

def normalize_ws(text):
    return " ".join(str(text).split()).strip()


def clean_answer(text):
    text = str(text)

    for marker in [
        "<end_of_turn>",
        "<start_of_turn>",
        "<|im_end|>",
        "<|im_start|>",
        "```",
    ]:
        text = text.replace(marker, "")

    if "Odgovor: " in text:
        text = text.split("Odgovor: ")[-1]

    return normalize_ws(text)


def safe_cer(true_text, pred_text):
    true_text = normalize_ws(true_text)
    pred_text = normalize_ws(pred_text)

    if not true_text and not pred_text:
        return 0.0

    if not true_text and pred_text:
        return 1.0

    if true_text and not pred_text:
        return 1.0

    return cer(true_text, pred_text)


def safe_wer(true_text, pred_text):
    true_text = normalize_ws(true_text)
    pred_text = normalize_ws(pred_text)

    if not true_text and not pred_text:
        return 0.0

    if not true_text and pred_text:
        return 1.0

    if true_text and not pred_text:
        return 1.0

    return wer(true_text, pred_text)


# ============================================================
# BOX EXTRACTION + IOU
# ============================================================

def extract_boxes(text):
    """
    Extract all [x1, y1, x2, y2] boxes from text.
    Keeps only numeric lists of length 4.
    """

    text = str(text)

    matches = re.findall(
        r"\[[^\[\]]+\]",
        text,
    )

    boxes = []

    for match in matches:
        try:
            box = ast.literal_eval(match)

            if (
                isinstance(box, list)
                and len(box) == 4
                and all(
                    isinstance(x, (int, float))
                    for x in box
                )
            ):
                boxes.append(
                    [float(x) for x in box]
                )

        except Exception:
            continue

    return boxes


def is_normalized_box(box):
    return (
        len(box) == 4
        and all(
            0.0 <= x <= 1.0
            for x in box
        )
    )


def calculate_iou(box_a, box_b):
    """
    Calculate IoU between two boxes:
    [x1, y1, x2, y2].
    """

    if len(box_a) != 4 or len(box_b) != 4:
        return 0.0

    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_area = (
        max(0.0, x_b - x_a)
        * max(0.0, y_b - y_a)
    )

    box_a_area = (
        max(0.0, box_a[2] - box_a[0])
        * max(0.0, box_a[3] - box_a[1])
    )

    box_b_area = (
        max(0.0, box_b[2] - box_b[0])
        * max(0.0, box_b[3] - box_b[1])
    )

    union_area = (
        box_a_area
        + box_b_area
        - inter_area
    )

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def greedy_match_iou(
    true_boxes,
    pred_boxes,
):
    """
    Order-independent matching.

    Each ground-truth box is matched to the unused
    predicted box with the highest IoU.

    Missing predictions receive IoU 0.
    Extra predictions are ignored.
    """

    if not true_boxes:
        return None

    used_pred = set()
    matched_ious = []

    for true_box in true_boxes:
        best_iou = 0.0
        best_j = None

        for j, pred_box in enumerate(
            pred_boxes
        ):
            if j in used_pred:
                continue

            iou = calculate_iou(
                true_box,
                pred_box,
            )

            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_j is not None:
            used_pred.add(best_j)

        matched_ious.append(
            best_iou
        )

    return float(
        np.mean(matched_ious)
    )


def greedy_match_iou_with_indices(
    true_boxes,
    pred_boxes,
):
    """
    Returns:
    - (true_idx, pred_idx, IoU) matches
    - matched IoUs
    - indices of used predictions
    """

    if not true_boxes:
        return [], [], set()

    used_pred = set()
    matches = []
    matched_ious = []

    for true_idx, true_box in enumerate(
        true_boxes
    ):
        best_iou = 0.0
        best_j = None

        for j, pred_box in enumerate(
            pred_boxes
        ):
            if j in used_pred:
                continue

            iou = calculate_iou(
                true_box,
                pred_box,
            )

            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_j is not None:
            used_pred.add(best_j)

            matches.append(
                (
                    true_idx,
                    best_j,
                    best_iou,
                )
            )

        else:
            matches.append(
                (
                    true_idx,
                    None,
                    0.0,
                )
            )

        matched_ious.append(
            best_iou
        )

    return (
        matches,
        matched_ious,
        used_pred,
    )


def detection_metrics_at_iou(
    true_boxes,
    pred_boxes,
    iou_threshold=0.5,
):
    """
    Detection metrics for one sample.

    A predicted box is a TP if it matches an unused
    ground-truth box with IoU >= threshold.
    """

    matched_gt = set()

    tp = 0
    fp = 0

    for pred_box in pred_boxes:
        best_iou = 0.0
        best_gt_idx = None

        for gt_idx, true_box in enumerate(
            true_boxes
        ):
            if gt_idx in matched_gt:
                continue

            iou = calculate_iou(
                true_box,
                pred_box,
            )

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if (
            best_iou >= iou_threshold
            and best_gt_idx is not None
        ):
            tp += 1
            matched_gt.add(
                best_gt_idx
            )

        else:
            fp += 1

    fn = (
        len(true_boxes)
        - len(matched_gt)
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# TEXT EXTRACTION
# ============================================================

def remove_boxes(text):
    return re.sub(
        r"\[[^\[\]]+\]",
        " ",
        str(text),
    )


def extract_page_text(text):
    """
    Extract text after 'Besedilo strani:'
    and before 'Slike:' if present.
    """

    text = str(text)

    if "Besedilo strani:" not in text:
        return ""

    part = text.split(
        "Besedilo strani:",
        1,
    )[1]

    if "Slike:" in part:
        part = part.split(
            "Slike:",
            1,
        )[0]

    return normalize_ws(part)


def extract_image_descriptions(text):
    """
    Extract descriptions after:
    Slika N opis:
    """

    text = str(text)

    pattern = (
        r"Slika\s+\d+\s+opis:\s*"
        r"(.*?)"
        r"(?="
        r"Slika\s+\d+\s+koordinate:"
        r"|Slika\s+\d+\s+opis:"
        r"|$)"
    )

    descs = re.findall(
        pattern,
        text,
        flags=re.DOTALL,
    )

    cleaned = []

    for desc in descs:
        desc = remove_boxes(desc)
        desc = normalize_ws(desc)

        if desc:
            cleaned.append(desc)

    return cleaned


def extract_main_text_for_dataset5(text):
    return clean_answer(text)


def greedy_match_text_error(
    true_texts,
    pred_texts,
    metric="wer",
):
    """
    Order-independent text matching.

    Missing predictions receive error 1.
    Extra predictions are ignored.
    """

    if not true_texts:
        return None

    used_pred = set()
    scores = []

    for true_text in true_texts:
        best_score = float("inf")
        best_j = None

        for j, pred_text in enumerate(
            pred_texts
        ):
            if j in used_pred:
                continue

            if metric == "cer":
                score = safe_cer(
                    true_text,
                    pred_text,
                )
            else:
                score = safe_wer(
                    true_text,
                    pred_text,
                )

            if score < best_score:
                best_score = score
                best_j = j

        if best_j is not None:
            used_pred.add(best_j)

        else:
            best_score = 1.0

        scores.append(best_score)

    return float(
        np.mean(scores)
    )


# ============================================================
# DATASET 1
# ============================================================

def evaluate_dataset1(
    rows,
    filepath=None,
):
    cers = []
    wers = []
    per_example = []

    for row in rows:
        true = clean_answer(
            row["true"]
        )

        pred = clean_answer(
            row["answer"]
        )

        cer_val = safe_cer(
            true,
            pred,
        )

        wer_val = safe_wer(
            true,
            pred,
        )

        cers.append(cer_val)
        wers.append(wer_val)

        row_dict = dict(row)
        row_dict["CER"] = cer_val
        row_dict["WER"] = wer_val

        per_example.append(
            row_dict
        )

    if filepath:
        pd.DataFrame(
            per_example
        ).to_csv(
            filepath.replace(
                ".tsv",
                "_per_example.csv",
            ),
            index=False,
        )

    return {
        "CER": (
            np.mean(cers)
            if cers
            else None
        ),
        "WER": (
            np.mean(wers)
            if wers
            else None
        ),
        "N": len(rows),
    }


# ============================================================
# DATASET 2
# ============================================================

def evaluate_dataset2(
    rows,
    filepath=None,
):
    page_ious = defaultdict(list)

    invalid_pred_boxes = 0
    non_normalized_pred_boxes = 0
    per_example = []

    for row in rows:
        true_boxes = extract_boxes(
            row["true"]
        )

        pred_boxes = extract_boxes(
            row["answer"]
        )

        image_path = row.get(
            "image_path",
            row.get(
                "id",
                "unknown",
            ),
        )

        is_invalid = 0

        if (
            true_boxes
            and not pred_boxes
        ):
            invalid_pred_boxes += 1
            is_invalid = 1

        is_non_norm = 0

        for box in pred_boxes:
            if not is_normalized_box(
                box
            ):
                non_normalized_pred_boxes += 1
                is_non_norm = 1

        score = greedy_match_iou(
            true_boxes,
            pred_boxes,
        )

        if score is not None:
            page_ious[
                image_path
            ].append(score)

        row_dict = dict(row)
        row_dict["IoU"] = score
        row_dict["Invalid"] = (
            is_invalid
        )
        row_dict["Non_normalized"] = (
            is_non_norm
        )

        per_example.append(
            row_dict
        )

    page_mean_ious = [
        float(np.mean(scores))
        for scores
        in page_ious.values()
        if scores
    ]

    if filepath:
        pd.DataFrame(
            per_example
        ).to_csv(
            filepath.replace(
                ".tsv",
                "_per_example.csv",
            ),
            index=False,
        )

    return {
        "Average IoU": (
            np.mean(page_mean_ious)
            if page_mean_ious
            else None
        ),
        "Invalid predictions":
            invalid_pred_boxes,
        "Non-normalized predicted boxes":
            non_normalized_pred_boxes,
        "N": len(rows),
        "N_pages": len(
            page_mean_ious
        ),
    }


# ============================================================
# DATASET 3
# ============================================================

def evaluate_dataset3(
    rows,
    filepath=None,
):
    ious = []
    precisions = []
    recalls = []
    f1s = []

    all_tp = 0
    all_fp = 0
    all_fn = 0

    invalid_pred_boxes = 0
    non_normalized_pred_boxes = 0
    per_example = []

    for row in rows:
        true_boxes = extract_boxes(
            row["true"]
        )

        pred_boxes = extract_boxes(
            row["answer"]
        )

        if (
            true_boxes
            and not pred_boxes
        ):
            invalid_pred_boxes += 1

        for box in pred_boxes:
            if not is_normalized_box(
                box
            ):
                non_normalized_pred_boxes += 1

        score = greedy_match_iou(
            true_boxes,
            pred_boxes,
        )

        if score is not None:
            ious.append(score)

        det = detection_metrics_at_iou(
            true_boxes,
            pred_boxes,
            iou_threshold=0.5,
        )

        precisions.append(
            det["precision"]
        )
        recalls.append(
            det["recall"]
        )
        f1s.append(
            det["f1"]
        )

        all_tp += det["TP"]
        all_fp += det["FP"]
        all_fn += det["FN"]

        row_dict = dict(row)

        row_dict["IoU"] = score
        row_dict["Precision@0.5"] = (
            det["precision"]
        )
        row_dict["Recall@0.5"] = (
            det["recall"]
        )
        row_dict["F1@0.5"] = (
            det["f1"]
        )
        row_dict["TP@0.5"] = (
            det["TP"]
        )
        row_dict["FP@0.5"] = (
            det["FP"]
        )
        row_dict["FN@0.5"] = (
            det["FN"]
        )

        per_example.append(
            row_dict
        )

    if filepath:
        pd.DataFrame(
            per_example
        ).to_csv(
            filepath.replace(
                ".tsv",
                "_per_example.csv",
            ),
            index=False,
        )

    return {
        "Average matched IoU": (
            np.mean(ious)
            if ious
            else None
        ),
        "Precision@0.5": (
            np.mean(precisions)
            if precisions
            else None
        ),
        "Recall@0.5": (
            np.mean(recalls)
            if recalls
            else None
        ),
        "F1@0.5": (
            np.mean(f1s)
            if f1s
            else None
        ),
        "TP@0.5": all_tp,
        "FP@0.5": all_fp,
        "FN@0.5": all_fn,
        "Invalid predictions":
            invalid_pred_boxes,
        "Non-normalized predicted boxes":
            non_normalized_pred_boxes,
        "N": len(rows),
    }


# ============================================================
# DATASET 4
# ============================================================

def evaluate_dataset4(
    rows,
    filepath=None,
):
    ious = []
    precisions = []
    recalls = []
    f1s = []

    all_tp = 0
    all_fp = 0
    all_fn = 0

    caption_cers = []
    caption_wers = []

    invalid_pred_boxes = 0
    non_normalized_pred_boxes = 0

    per_example = []

    for row in rows:
        true = row["true"]
        pred = row["answer"]

        true_boxes = extract_boxes(
            true
        )

        pred_boxes = extract_boxes(
            pred
        )

        true_descs = (
            extract_image_descriptions(
                true
            )
        )

        pred_descs = (
            extract_image_descriptions(
                pred
            )
        )

        if (
            true_boxes
            and not pred_boxes
        ):
            invalid_pred_boxes += 1

        for box in pred_boxes:
            if not is_normalized_box(
                box
            ):
                non_normalized_pred_boxes += 1

        (
            matches,
            matched_ious,
            _,
        ) = greedy_match_iou_with_indices(
            true_boxes,
            pred_boxes,
        )

        ious.extend(
            matched_ious
        )

        det = detection_metrics_at_iou(
            true_boxes,
            pred_boxes,
            iou_threshold=0.5,
        )

        precisions.append(
            det["precision"]
        )
        recalls.append(
            det["recall"]
        )
        f1s.append(
            det["f1"]
        )

        all_tp += det["TP"]
        all_fp += det["FP"]
        all_fn += det["FN"]

        per_row_caption_cers = []
        per_row_caption_wers = []

        for (
            true_idx,
            pred_idx,
            box_iou,
        ) in matches:

            true_caption = (
                true_descs[true_idx]
                if true_idx
                < len(true_descs)
                else ""
            )

            pred_caption = (
                pred_descs[pred_idx]
                if (
                    pred_idx is not None
                    and pred_idx
                    < len(pred_descs)
                )
                else ""
            )

            if (
                pred_idx is not None
                and box_iou >= 0.5
            ):
                cer_score = safe_cer(
                    true_caption,
                    pred_caption,
                )

                wer_score = safe_wer(
                    true_caption,
                    pred_caption,
                )

            else:
                cer_score = 1.0
                wer_score = 1.0

            caption_cers.append(
                cer_score
            )

            caption_wers.append(
                wer_score
            )

            per_row_caption_cers.append(
                cer_score
            )

            per_row_caption_wers.append(
                wer_score
            )

        if filepath is not None:
            row_dict = dict(row)

            row_dict["IoU"] = (
                float(
                    np.mean(
                        matched_ious
                    )
                )
                if matched_ious
                else None
            )

            row_dict["Precision@0.5"] = (
                det["precision"]
            )

            row_dict["Recall@0.5"] = (
                det["recall"]
            )

            row_dict["F1@0.5"] = (
                det["f1"]
            )

            row_dict["TP@0.5"] = (
                det["TP"]
            )

            row_dict["FP@0.5"] = (
                det["FP"]
            )

            row_dict["FN@0.5"] = (
                det["FN"]
            )

            row_dict["Caption_CER"] = (
                float(
                    np.mean(
                        per_row_caption_cers
                    )
                )
                if per_row_caption_cers
                else None
            )

            row_dict["Caption_WER"] = (
                float(
                    np.mean(
                        per_row_caption_wers
                    )
                )
                if per_row_caption_wers
                else None
            )

            per_example.append(
                row_dict
            )

    if filepath is not None:
        pd.DataFrame(
            per_example
        ).to_csv(
            filepath.replace(
                ".tsv",
                "_per_example.csv",
            ),
            index=False,
        )

    return {
        "Average matched IoU": (
            np.mean(ious)
            if ious
            else None
        ),
        "Precision@0.5": (
            np.mean(precisions)
            if precisions
            else None
        ),
        "Recall@0.5": (
            np.mean(recalls)
            if recalls
            else None
        ),
        "F1@0.5": (
            np.mean(f1s)
            if f1s
            else None
        ),
        "TP@0.5": all_tp,
        "FP@0.5": all_fp,
        "FN@0.5": all_fn,
        "Caption CER": (
            np.mean(caption_cers)
            if caption_cers
            else None
        ),
        "Caption WER": (
            np.mean(caption_wers)
            if caption_wers
            else None
        ),
        "Invalid predictions":
            invalid_pred_boxes,
        "Non-normalized predicted boxes":
            non_normalized_pred_boxes,
        "N": len(rows),
    }


# ============================================================
# DATASET 5
# ============================================================

def evaluate_dataset5(
    rows,
    filepath=None,
):
    cers = []
    wers = []
    per_example = []

    for row in rows:
        true = (
            extract_main_text_for_dataset5(
                row["true"]
            )
        )

        pred = (
            extract_main_text_for_dataset5(
                row["answer"]
            )
        )

        cer_val = safe_cer(
            true,
            pred,
        )

        wer_val = safe_wer(
            true,
            pred,
        )

        cers.append(cer_val)
        wers.append(wer_val)

        row_dict = dict(row)
        row_dict["CER"] = cer_val
        row_dict["WER"] = wer_val

        per_example.append(
            row_dict
        )

    if filepath is not None:
        pd.DataFrame(
            per_example
        ).to_csv(
            filepath.replace(
                ".tsv",
                "_per_example.csv",
            ),
            index=False,
        )

    return {
        "CER": (
            np.mean(cers)
            if cers
            else None
        ),
        "WER": (
            np.mean(wers)
            if wers
            else None
        ),
        "N": len(rows),
    }


# ============================================================
# DATASET 6
# ============================================================

def evaluate_dataset6(
    rows,
    filepath=None,
):
    page_cers = []
    page_wers = []

    ious = []
    precisions = []
    recalls = []
    f1s = []

    all_tp = 0
    all_fp = 0
    all_fn = 0

    caption_cers = []
    caption_wers = []

    invalid_pred_boxes = 0
    non_normalized_pred_boxes = 0

    rows_with_page_text = 0
    rows_with_page_section_eval = 0
    false_page_sections = 0
    missing_page_sections = 0
    rows_with_boxes = 0
    rows_with_captions = 0

    per_example = []

    for row in rows:
        true = row["true"]
        pred = row["answer"]

        page_cer = None
        page_wer = None
        iou_score = None
        caption_cer_score = None
        caption_wer_score = None

        # -------------------------
        # PAGE TEXT
        # -------------------------

        true_has_page_section = (
            "Besedilo strani:"
            in str(true)
        )

        pred_has_page_section = (
            "Besedilo strani:"
            in str(pred)
        )

        true_page_text = (
            extract_page_text(true)
        )

        pred_page_text = (
            extract_page_text(pred)
        )

        if true_has_page_section:
            rows_with_page_text += 1

        if (
            true_has_page_section
            or pred_has_page_section
        ):
            rows_with_page_section_eval += 1

            page_cer = safe_cer(
                true_page_text,
                pred_page_text,
            )

            page_wer = safe_wer(
                true_page_text,
                pred_page_text,
            )

            page_cers.append(
                page_cer
            )

            page_wers.append(
                page_wer
            )

        if (
            not true_has_page_section
            and pred_has_page_section
        ):
            false_page_sections += 1

        if (
            true_has_page_section
            and not pred_has_page_section
        ):
            missing_page_sections += 1

        # -------------------------
        # BOXES
        # -------------------------

        true_boxes = extract_boxes(
            true
        )

        pred_boxes = extract_boxes(
            pred
        )

        true_descs = (
            extract_image_descriptions(
                true
            )
        )

        pred_descs = (
            extract_image_descriptions(
                pred
            )
        )

        if true_boxes:
            rows_with_boxes += 1

        if (
            true_boxes
            and not pred_boxes
        ):
            invalid_pred_boxes += 1

        for box in pred_boxes:
            if not is_normalized_box(
                box
            ):
                non_normalized_pred_boxes += 1

        (
            matches,
            matched_ious,
            _,
        ) = greedy_match_iou_with_indices(
            true_boxes,
            pred_boxes,
        )

        if matched_ious:
            iou_score = float(
                np.mean(
                    matched_ious
                )
            )

            ious.extend(
                matched_ious
            )

        det = detection_metrics_at_iou(
            true_boxes,
            pred_boxes,
            iou_threshold=0.5,
        )

        precisions.append(
            det["precision"]
        )

        recalls.append(
            det["recall"]
        )

        f1s.append(
            det["f1"]
        )

        all_tp += det["TP"]
        all_fp += det["FP"]
        all_fn += det["FN"]

        # -------------------------
        # CAPTIONS
        # -------------------------

        per_row_caption_cers = []
        per_row_caption_wers = []

        if true_descs:
            rows_with_captions += 1

            if true_boxes:
                for (
                    true_idx,
                    pred_idx,
                    box_iou,
                ) in matches:

                    true_caption = (
                        true_descs[true_idx]
                        if true_idx
                        < len(true_descs)
                        else ""
                    )

                    pred_caption = (
                        pred_descs[pred_idx]
                        if (
                            pred_idx is not None
                            and pred_idx
                            < len(pred_descs)
                        )
                        else ""
                    )

                    if true_caption:
                        if (
                            pred_idx
                            is not None
                            and box_iou
                            >= 0.5
                        ):
                            c = safe_cer(
                                true_caption,
                                pred_caption,
                            )

                            w = safe_wer(
                                true_caption,
                                pred_caption,
                            )

                        else:
                            c = 1.0
                            w = 1.0

                        caption_cers.append(
                            c
                        )

                        caption_wers.append(
                            w
                        )

                        per_row_caption_cers.append(
                            c
                        )

                        per_row_caption_wers.append(
                            w
                        )

            else:
                c = greedy_match_text_error(
                    true_descs,
                    pred_descs,
                    metric="cer",
                )

                w = greedy_match_text_error(
                    true_descs,
                    pred_descs,
                    metric="wer",
                )

                if c is not None:
                    caption_cers.append(c)
                    per_row_caption_cers.append(c)

                if w is not None:
                    caption_wers.append(w)
                    per_row_caption_wers.append(w)

        if per_row_caption_cers:
            caption_cer_score = float(
                np.mean(
                    per_row_caption_cers
                )
            )

        if per_row_caption_wers:
            caption_wer_score = float(
                np.mean(
                    per_row_caption_wers
                )
            )

        # -------------------------
        # PER-EXAMPLE OUTPUT
        # -------------------------

        if filepath is not None:
            row_dict = dict(row)

            row_dict["Page_CER"] = (
                page_cer
            )

            row_dict["Page_WER"] = (
                page_wer
            )

            row_dict[
                "GT_has_Besedilo_strani"
            ] = true_has_page_section

            row_dict[
                "Pred_has_Besedilo_strani"
            ] = pred_has_page_section

            row_dict[
                "False_page_section"
            ] = int(
                (
                    not true_has_page_section
                )
                and pred_has_page_section
            )

            row_dict[
                "Missing_page_section"
            ] = int(
                true_has_page_section
                and (
                    not pred_has_page_section
                )
            )

            row_dict["IoU"] = (
                iou_score
            )

            row_dict[
                "Precision@0.5"
            ] = det["precision"]

            row_dict[
                "Recall@0.5"
            ] = det["recall"]

            row_dict[
                "F1@0.5"
            ] = det["f1"]

            row_dict[
                "TP@0.5"
            ] = det["TP"]

            row_dict[
                "FP@0.5"
            ] = det["FP"]

            row_dict[
                "FN@0.5"
            ] = det["FN"]

            row_dict[
                "Caption_CER"
            ] = caption_cer_score

            row_dict[
                "Caption_WER"
            ] = caption_wer_score

            per_example.append(
                row_dict
            )

    if filepath is not None:
        pd.DataFrame(
            per_example
        ).to_csv(
            filepath.replace(
                ".tsv",
                "_per_example.csv",
            ),
            index=False,
        )

    return {
        "Page CER": (
            np.mean(page_cers)
            if page_cers
            else None
        ),
        "Page WER": (
            np.mean(page_wers)
            if page_wers
            else None
        ),
        "Rows with GT page text":
            rows_with_page_text,
        "Rows evaluated for page section":
            rows_with_page_section_eval,
        "False page sections":
            false_page_sections,
        "Missing page sections":
            missing_page_sections,
        "Average matched IoU": (
            np.mean(ious)
            if ious
            else None
        ),
        "Precision@0.5": (
            np.mean(precisions)
            if precisions
            else None
        ),
        "Recall@0.5": (
            np.mean(recalls)
            if recalls
            else None
        ),
        "F1@0.5": (
            np.mean(f1s)
            if f1s
            else None
        ),
        "TP@0.5": all_tp,
        "FP@0.5": all_fp,
        "FN@0.5": all_fn,
        "Caption CER": (
            np.mean(caption_cers)
            if caption_cers
            else None
        ),
        "Caption WER": (
            np.mean(caption_wers)
            if caption_wers
            else None
        ),
        "Invalid box predictions":
            invalid_pred_boxes,
        "Non-normalized predicted boxes":
            non_normalized_pred_boxes,
        "Rows with boxes":
            rows_with_boxes,
        "Rows with captions":
            rows_with_captions,
        "N": len(rows),
    }


# ============================================================
# FILE ROUTING
# ============================================================

def evaluate_file(filepath):
    filename = os.path.basename(
        filepath
    )

    rows = []

    with open(
        filepath,
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(
            file,
            delimiter="\t",
        )

        for row in reader:
            rows.append(row)

    if "dataset1_bb_2_caption" in filename:
        return evaluate_dataset1(
            rows,
            filepath,
        )

    elif "dataset2_caption_2_bb" in filename:
        return evaluate_dataset2(
            rows,
            filepath,
        )

    elif "dataset3_all_bb" in filename:
        return evaluate_dataset3(
            rows,
            filepath,
        )

    elif "dataset4_all_bb_and_captions2" in filename:
        return evaluate_dataset4(
            rows,
            filepath,
        )

    elif "dataset5_metatext" in filename:
        return evaluate_dataset5(
            rows,
            filepath,
        )

    elif "dataset6_retrieve_all2" in filename:
        return evaluate_dataset6(
            rows,
            filepath,
        )

    else:
        print(
            f"Skipping unknown file: "
            f"{filename}"
        )

        return None


def detect_dataset_name(filename):
    dataset_names = [
        "dataset1_bb_2_caption",
        "dataset2_caption_2_bb",
        "dataset3_all_bb",
        "dataset4_all_bb_and_captions2",
        "dataset5_metatext",
        "dataset6_retrieve_all2",
    ]

    for name in dataset_names:
        if name in filename:
            return name

    return "unknown"


def detect_evaluation_mode(path):
    path = path.lower()

    if (
        "oversampling" in path
        or "_os" in path
    ):
        return "sft_oversampling"

    if "zero_shot" in path:
        return "zero_shot"

    if "one_shot" in path:
        return "one_shot"

    if "sft" in path:
        return "sft"

    return "unknown"


# ============================================================
# MAIN
# ============================================================

def main():
    records = []

    for result_dir in RESULT_DIRS:
        if not os.path.isdir(
            result_dir
        ):
            print(
                f"Missing result directory, "
                f"skipping: {result_dir}"
            )
            continue

        for filename in sorted(
            os.listdir(result_dir)
        ):
            if not filename.endswith(
                ".tsv"
            ):
                continue

            filepath = os.path.join(
                result_dir,
                filename,
            )

            print(
                f"Evaluating: "
                f"{filepath}"
            )

            scores = evaluate_file(
                filepath
            )

            if scores is None:
                continue

            dataset_name = (
                detect_dataset_name(
                    filename
                )
            )

            evaluation_mode = (
                detect_evaluation_mode(
                    filepath
                )
            )

            record = {
                "result_dir":
                    result_dir,
                "file":
                    filename,
                "dataset":
                    dataset_name,
                "evaluation_mode":
                    evaluation_mode,
            }

            record.update(
                scores
            )

            records.append(
                record
            )

    df = pd.DataFrame(
        records
    )

    print(
        "\n================ "
        "FINAL RESULTS "
        "================"
    )

    print(df)

    df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8",
    )

    print(
        f"\nSaved: "
        f"{OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()