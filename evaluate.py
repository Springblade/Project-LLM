import os
import json
from pathlib import Path
from config import (
    GOOGLE_API_KEY,
    GEMINI_MODEL,
    TEMPERATURE,
    DATASET_PATH,
    RESULTS_OUTPUT_PATH
)

from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    AnswerRelevancy,
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from add_image_caption import images_metadata


class RagasEvaluator:
    def __init__(self, dataset_path: str):
        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        self.dataset_path = dataset_path
        self.dataset = None
        self.llm = None
        self.embeddings = None

    def setup_gemini(self):
        """Configure Gemini LLM and embeddings"""
        print("🔧 Configuring Gemini LLM...")
        self.llm = LangchainLLMWrapper(
            ChatGoogleGenerativeAI(
                model=GEMINI_MODEL,
                temperature=TEMPERATURE,
                max_tokens=2048
            )
        )
        print("🔧 Configuring Gemini Embeddings...")
        self.embeddings = LangchainEmbeddingsWrapper(
            GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                task_type="retrieval_document"
            )
        )
        print("✅ Gemini setup complete!\n")

    def load_dataset(self):
        """Load dataset and append image captions to question"""
        print(f"📂 Loading dataset from {self.dataset_path}...")
        if not Path(self.dataset_path).exists():
            raise FileNotFoundError(f"File not found: {self.dataset_path}")

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        dataset_list = []
        for case_id, case_data in raw_data.items():
            question_item = case_data["question"]
            question_text = question_item.get("Text", "")

            # Parse image list
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

            # Combine question text + captions
            full_question = question_text
            if captions:
                full_question += "\n\n" + "\n".join([f"[Image Caption]: {c}" for c in captions])
            contexts = case_data.get("contexts")
            if isinstance(contexts, str):
                contexts = [contexts] if contexts else []
            elif contexts is None:
                contexts = []
            dataset_list.append({
                "user_input": full_question,
                "response": case_data["answer"],        # Predicted answer
                "retrieved_contexts": contexts, # get contexts if any
                "reference": case_data["ground_truth"] # Ground truth reference
            })

        self.dataset = EvaluationDataset.from_list(dataset_list)
        print(f"✅ Loaded {len(self.dataset)} samples with image captions\n")

    def evaluate(self):
        if self.llm is None or self.embeddings is None:
            self.setup_gemini()

        if self.dataset is None:
            self.load_dataset()

        metrics = [
            ContextPrecision(llm=self.llm),
            ContextRecall(llm=self.llm),
            Faithfulness(llm=self.llm),
            AnswerRelevancy(llm=self.llm),
        ]  

        print("⏳ Running evaluation...\n")

        results = evaluate(
            dataset=self.dataset,
            metrics=metrics,
            llm=self.llm,
            embeddings=self.embeddings
        )
        return results

    def save_results(self, results, output_path: str = RESULTS_OUTPUT_PATH):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df_results = results.to_pandas()
        df_results.to_csv(output_path, index=False)
        print(f"✅ Results saved to {output_path}\n")
        return df_results

    def display_results(self, df_results):
        print("=" * 80)
        print("📊 EVALUATION RESULTS")
        print("=" * 80)
        print(df_results)
        print("\n" + "=" * 80)
        print("📈 SUMMARY STATISTICS")
        print("=" * 80)
        print(df_results.describe())
        print("\n")


def main():
    evaluator = RagasEvaluator(dataset_path=DATASET_PATH)
    evaluator.setup_gemini()
    evaluator.load_dataset()
    results = evaluator.evaluate()
    df_results = evaluator.save_results(results)
    evaluator.display_results(df_results)
    print("✅ Evaluation completed successfully!")


if __name__ == "__main__":
    main()
