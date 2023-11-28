import os
import json
import argparse
import numpy as np

from metrics import (
    qa_f1_score,
    rouge_score,
    classification_score,
    retrieval_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

oneliners = ["narrativeqa", "qasper", "multifieldqa_en", 
             "hotpotqa", "2wikimqa", "musique", "trec", 
             "triviaqa", "samsum", "passage_count", 
             "passage_retrieval_en"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")
    return parser.parse_args()


def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for prediction, ground_truths, length in zip(predictions, answers, lengths):
        score = 0.0
        if dataset in oneliners:
            prediction = prediction.lstrip("\n").split("\n")[0]
        for ground_truth in ground_truths:
            score = max(
                score,
                dataset2metric[dataset](
                    prediction, ground_truth, all_classes=all_classes
                ),
            )
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores


def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.0
    for prediction, ground_truths in zip(predictions, answers):
        score = 0.0
        if dataset in oneliners:
            prediction = prediction.lstrip("\n").split("\n")[0]
        for ground_truth in ground_truths:
            score = max(
                score,
                dataset2metric[dataset](
                    prediction, ground_truth, all_classes=all_classes
                ),
            )
        total_score += score
    return round(100 * total_score / len(predictions), 2)


def process_file(path, filename, args):
    predictions, answers, lengths, all_classes = [], [], [], None
    with open(os.path.join(path, filename), "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            predictions.append(data["pred"])
            answers.append(data["answers"])
            all_classes = data.get("all_classes")
            lengths.append(data.get("length", 0))

    dataset = filename.split(".")[0]
    if args.e:
        return dataset, scorer_e(dataset, predictions, answers, lengths, all_classes)
    else:
        return dataset, scorer(dataset, predictions, answers, all_classes)

def main():
    args = parse_args()
    scores = {}

    base_path = "pred_e" if args.e else "pred"
    model_path = os.path.join(base_path, args.model)
    all_files = [f for f in os.listdir(model_path) if f.endswith(".jsonl")]
    
    print("Evaluating on:", all_files)
    for filename in all_files:
        dataset, score = process_file(model_path, filename, args)
        scores[dataset] = score

    output_file = os.path.join(model_path, "result.json")
    with open(output_file, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)

    print("Scores:", json.dumps(scores, indent=4))
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()