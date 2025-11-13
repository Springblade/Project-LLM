import os
import json
import time
from pathlib import Path
import pandas as pd
from nltk.tokenize import word_tokenize
from evaluate import load
from config import DATASET_PATH, RESULTS_OUTPUT_PATH
from add_image_caption import METADATA_PATH, images_metadata
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")


class SimpleEvaluator:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.dataset = None
        self.results = None
        self.bertscore = load("bertscore")

    def load_dataset(self):
        """Load dataset and integrate image captions (same logic as RagasEvaluator)"""
        print(f"📂 Loading dataset from {self.dataset_path}...")

        if not Path(self.dataset_path).exists():
            raise FileNotFoundError(f"File not found: {self.dataset_path}")

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        dataset_list = []

        for case_id, case_data in raw_data.items():
            question_item = case_data["question"]
            question_text = question_item.get("Text", "")

            # Parse image list safely
            image_list = []
            if image_list_raw := question_item.get("ImageList"):
                if isinstance(image_list_raw, str):
                    try:
                        image_list = json.loads(image_list_raw)
                    except Exception:
                        image_list = []
                elif isinstance(image_list_raw, list):
                    image_list = image_list_raw

            # Collect captions
            captions = []
            for img in image_list:
                if img in images_metadata:
                    captions.append(images_metadata[img]["Caption"])
                else:
                    print(f"⚠️ No caption found for image: {img}")

            # Combine question text with captions
            full_question = question_text
            if captions:
                full_question += "\n\n" + "\n".join([f"[Image Caption]: {c}" for c in captions])

            dataset_list.append({
                "question": full_question,
                "predicted": case_data["answer"],
                "reference": case_data["ground_truth"]
            })

        self.dataset = dataset_list
        print(f"✅ Loaded {len(self.dataset)} samples with integrated captions\n")

    @staticmethod
    def compute_f1(pred, ref):
        """Compute token-level F1 between predicted and reference answers"""
        pred_tokens = set(word_tokenize(pred.lower()))
        ref_tokens = set(word_tokenize(ref.lower()))
        if not pred_tokens or not ref_tokens:
            return 0.0
        tp = len(pred_tokens & ref_tokens)
        precision = tp / len(pred_tokens)
        recall = tp / len(ref_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def evaluate(self):
        """Compute F1 and BERTScore metrics"""
        if self.dataset is None:
            self.load_dataset()

        preds = [d["predicted"] for d in self.dataset]
        refs = [d["reference"] for d in self.dataset]

        print("⏳ Calculating BERTScore (semantic similarity)...")
        bert_results = self.bertscore.compute(predictions=preds, references=refs, lang="en")
        print("✅ BERTScore completed\n")

        f1_scores = [self.compute_f1(p, r) for p, r in zip(preds, refs)]

        self.results = pd.DataFrame({
            "Question": [d["question"] for d in self.dataset],
            "Predicted_Answer": preds,
            "Reference_Answer": refs,
            "F1_Score": f1_scores,
            "BERTScore_Precision": bert_results["precision"],
            "BERTScore_Recall": bert_results["recall"],
            "BERTScore_F1": bert_results["f1"],
        })

        return self.results

    def save_results(self, output_path: str = RESULTS_OUTPUT_PATH):
        """Save results to CSV"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self.results.to_csv(output_path, index=False)
        print(f"✅ Results saved to {output_path}\n")

    def display_results(self):
        """Display results summary"""
        print("=" * 100)
        print("📊 EVALUATION RESULTS")
        print("=" * 100)
        print(self.results.head())
        print("\n" + "=" * 100)
        print("📈 SUMMARY STATISTICS")
        print("=" * 100)
        print(self.results.describe())
        print("\n")


def main():
    evaluator = SimpleEvaluator(dataset_path=DATASET_PATH)
    evaluator.load_dataset()
    df_results = evaluator.evaluate()
    evaluator.save_results()
    evaluator.display_results()


if __name__ == "__main__":
    main()
