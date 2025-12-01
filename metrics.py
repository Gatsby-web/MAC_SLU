import json
import argparse
import sys
import re
from collections import defaultdict

def normalize_text(text):
    """
    Normalizes text:
    1. Convert to lowercase.
    2. Convert Chinese numerals to Arabic numerals (character-level replacement).
    3. Remove punctuation.
    4. Remove extra whitespace.
    """
    if not isinstance(text, str):
        return str(text)

    # 1. Convert to lowercase
    text = text.lower()

    # 2. Chinese numeral mapping (Simple character replacement)
    # KEEPING CHINESE CHARACTERS HERE AS REQUESTED
    cn_num_map = {
        '零': '0', '一': '1', '二': '2', '三': '3', '四': '4',
        '五': '5', '六': '6', '七': '7', '八': '8', '九': '9',
        '两': '2'
    }
    for k, v in cn_num_map.items():
        text = text.replace(k, v)

    # 3. Remove punctuation
    # Logic: Replace any character that is NOT a word char, digit, or whitespace with empty string.
    # \w in Python 3 re includes alphanumeric characters (including Chinese characters) and underscores.
    text = re.sub(r'[^\w\s]', '', text)

    # 4. Remove leading/trailing whitespace
    return text.strip()

def normalize_semantics(semantics_list):
    """
    Recursively normalizes all key fields in the semantics list.
    """
    if not isinstance(semantics_list, list):
        return []

    normalized_list = []
    for item in semantics_list:
        if not isinstance(item, dict):
            continue
            
        new_item = {}
        
        # Normalize domain and intent
        if 'domain' in item:
            new_item['domain'] = normalize_text(item['domain'])
        if 'intent' in item:
            new_item['intent'] = normalize_text(item['intent'])
            
        # Normalize slots
        if 'slots' in item:
            origin_slots = item['slots']
            if isinstance(origin_slots, dict):
                new_slots = {}
                for k, v in origin_slots.items():
                    # Normalize both Slot Key and Value
                    norm_k = normalize_text(k)
                    norm_v = normalize_text(v)
                    new_slots[norm_k] = norm_v
                new_item['slots'] = new_slots
            else:
                new_item['slots'] = origin_slots
        
        normalized_list.append(new_item)
    
    return normalized_list

def calculate_metrics(predict_file, ground_truth_file):
    """
    Reads prediction and ground truth files line by line to calculate multiple evaluation metrics.
    Data is normalized before comparison.

    Args:
        predict_file (str): Path to the prediction JSONL file.
        ground_truth_file (str): Path to the ground truth JSONL file.

    Returns:
        dict: A dictionary containing the evaluation results.
    """
    try:
        with open(predict_file, 'r', encoding='utf-8') as f_pred:
            predict_lines = f_pred.readlines()
        with open(ground_truth_file, 'r', encoding='utf-8') as f_gt:
            ground_truth_lines = f_gt.readlines()
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        sys.exit(1)

    if len(predict_lines) != len(ground_truth_lines):
        print(
            f"Error: File line counts do not match.\n"
            f"  - Prediction File ({predict_file}): {len(predict_lines)} lines\n"
            f"  - Ground Truth File ({ground_truth_file}): {len(ground_truth_lines)} lines",
            file=sys.stderr
        )
        sys.exit(1)

    total_count = len(predict_lines)
    if total_count == 0:
        print("Warning: File is empty, cannot calculate metrics.", file=sys.stderr)
        return {
            "total_count": 0, "overall_match_count": 0, "overall_accuracy": 1.0,
            "intent_match_count": 0, "intent_accuracy": 1.0,
            "slot_tp": 0, "slot_fp": 0, "slot_fn": 0,
            "slot_precision": 1.0, "slot_recall": 1.0, "slot_f1": 1.0,
        }

    # Initialize counters
    overall_match_count = 0
    intent_match_count = 0
    slot_tp, slot_fp, slot_fn = 0, 0, 0 

    for i, (pred_line, gt_line) in enumerate(zip(predict_lines, ground_truth_lines)):
        try:
            pred_data = json.loads(pred_line.strip())
            gt_data = json.loads(gt_line.strip())

            # Get raw semantics
            raw_pred_semantics = pred_data.get("semantics", [])
            raw_gt_semantics = gt_data.get("semantics", [])

            # --- [New Step] Data Normalization ---
            pred_semantics = normalize_semantics(raw_pred_semantics)
            gt_semantics = normalize_semantics(raw_gt_semantics)
            
            # --- 1. Calculate Overall Accuracy (Exact Match) ---
            # Requires the entire structure (after normalization) to be identical.
            # Note: Different list orders will result in a mismatch.
            if pred_semantics == gt_semantics:
                overall_match_count += 1
            
            # --- 2. Calculate Intent Accuracy (All Intents Correct) ---
            # Extract all (domain, intent) pairs in the sample.
            # Use sorted() to ignore the list order.
            pred_intents = sorted([(s.get("domain"), s.get("intent")) for s in pred_semantics])
            gt_intents = sorted([(s.get("domain"), s.get("intent")) for s in gt_semantics])

            if pred_intents == gt_intents:
                intent_match_count += 1

            # --- 3. Calculate Slot Filling Metrics (Global Aggregation) ---
            pred_slot_set = set()
            for s in pred_semantics:
                slots = s.get("slots", {})
                if isinstance(slots, dict):
                    for k, v in slots.items():
                        pred_slot_set.add((k, v))
            
            gt_slot_set = set()
            for s in gt_semantics:
                slots = s.get("slots", {})
                if isinstance(slots, dict):
                    for k, v in slots.items():
                        gt_slot_set.add((k, v))

            # Calculate TP, FP, FN
            slot_tp += len(pred_slot_set.intersection(gt_slot_set))
            slot_fp += len(pred_slot_set.difference(gt_slot_set))
            slot_fn += len(gt_slot_set.difference(pred_slot_set))

        except json.JSONDecodeError:
            print(f"Warning: JSON parse error at line {i+1}, skipped.", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Unknown error at line {i+1}: {e}, skipped.", file=sys.stderr)

    # --- Calculate Final Metrics ---
    overall_accuracy = overall_match_count / total_count if total_count > 0 else 0.0
    intent_accuracy = intent_match_count / total_count if total_count > 0 else 0.0

    # Calculate Slot F1
    if slot_tp + slot_fp == 0:
        slot_precision = 1.0 if slot_fn == 0 else 0.0
    else:
        slot_precision = slot_tp / (slot_tp + slot_fp)

    if slot_tp + slot_fn == 0:
        slot_recall = 1.0 if slot_fp == 0 else 0.0
    else:
        slot_recall = slot_tp / (slot_tp + slot_fn)

    if slot_precision + slot_recall == 0:
        slot_f1 = 0.0
    else:
        slot_f1 = 2 * (slot_precision * slot_recall) / (slot_precision + slot_recall)

    return {
        "total_count": total_count,
        "overall_match_count": overall_match_count,
        "overall_accuracy": overall_accuracy,
        "intent_match_count": intent_match_count,
        "intent_accuracy": intent_accuracy,
        "slot_tp": slot_tp,
        "slot_fp": slot_fp,
        "slot_fn": slot_fn,
        "slot_precision": slot_precision,
        "slot_recall": slot_recall,
        "slot_f1": slot_f1,
    }

def main():
    parser = argparse.ArgumentParser(
        description="Calculate NLU Evaluation Metrics (Multi-intent & Normalization supported)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("predict_file", help="Path to prediction .jsonl file")
    parser.add_argument("ground_truth_file", help="Path to ground truth .jsonl file")
    args = parser.parse_args()

    results = calculate_metrics(args.predict_file, args.ground_truth_file)

    print("-" * 60)
    print(f"Evaluation Results (Normalization Enabled: Case/Punct/Num)")
    print("-" * 60)
    print(f"Total Records Processed: {results['total_count']}")
    
    print("\n--- Overall Accuracy (Semantics Exact Match) ---")
    print(f"Exact Matches:   {results['overall_match_count']}")
    print(f"Accuracy:        {results['overall_accuracy']:.4f} ({results['overall_accuracy']:.2%})")
    
    print("\n--- Intent Accuracy (All Intents Correct) ---")
    print(f"Intent Matches:  {results['intent_match_count']}")
    print(f"Accuracy:        {results['intent_accuracy']:.4f} ({results['intent_accuracy']:.2%})")

    print("\n--- Slot Filling F1-Score (Global Aggregation) ---")
    print(f"TP / FP / FN:    {results['slot_tp']} / {results['slot_fp']} / {results['slot_fn']}")
    print(f"Precision:       {results['slot_precision']:.4f}")
    print(f"Recall:          {results['slot_recall']:.4f}")
    print(f"F1 Score:        {results['slot_f1']:.4f}")
    print("-" * 60)

if __name__ == "__main__":
    main()