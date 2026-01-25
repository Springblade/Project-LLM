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

from ragas.metrics import AnswerRelevancy
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


class RagasEvaluator:
    def __init__(self, dataset_path: str):
        """Initialize Ragas Evaluator with Gemini API"""
        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        self.dataset_path = dataset_path
        self.llm = None
        self.embeddings = None
        self.dataset = None
        
    def setup_gemini(self):
        """Configure Gemini LLM and Embeddings"""
        print("🔧 Configuring Gemini LLM...")
        
        self.llm = LangchainLLMWrapper(
            ChatGoogleGenerativeAI(
                model=GEMINI_MODEL,
                temperature=TEMPERATURE,
                max_tokens=2048,
            )
        )
        
        print("🔧 Configuring Gemini Embeddings...")
        
        # ✅ Using LangchainEmbeddingsWrapper
        self.embeddings = LangchainEmbeddingsWrapper(
            GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                task_type="retrieval_document"
            )
        )
        print("✅ Gemini configuration completed!\n")
    
    def load_dataset(self):
        """Load dataset from JSON file"""
        print(f"📂 Loading dataset from {self.dataset_path}...")
        
        # ✅ Check if file exists
        if not Path(self.dataset_path).exists():
            raise FileNotFoundError(f"File not found: {self.dataset_path}")
        
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        dataset_list = [
            {
                "user_input": item["question"],
                "response": item["answer"],
                "retrieved_contexts": item["contexts"],
                "reference": item["ground_truth"]
            }
            for item in raw_data.values()
        ]
        
        self.dataset = EvaluationDataset.from_list(dataset_list)
        print(f"✅ Loaded {len(self.dataset)} samples\n")
    
    def evaluate(self):  # ✅ NOT async def
        """Run evaluation with Ragas"""
        if self.llm is None or self.embeddings is None:
            self.setup_gemini()
        
        if self.dataset is None:
            self.load_dataset()

        # Define metrics
        metrics = [
            ContextPrecision(llm=self.llm),
            ContextRecall(llm=self.llm),
            Faithfulness(llm=self.llm),
            AnswerRelevancy(llm=self.llm),
        ]     
        
        print("⏳ Running evaluation (this may take a few minutes)...\n")
        
        # ✅ No await - call directly
        results = evaluate(
            dataset=self.dataset,
            metrics=metrics,
            llm=self.llm,
            embeddings=self.embeddings
        )
        
        return results
    
    def save_results(self, results, output_path: str = RESULTS_OUTPUT_PATH):
        """Save evaluation results"""
        # Create directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        df_results = results.to_pandas()
        df_results.to_csv(output_path, index=False)
        print(f"✅ Results saved to {output_path}\n")
        
        return df_results
    
    def display_results(self, df_results):
        """Display results"""
        print("=" * 80)
        print("📊 EVALUATION RESULTS")
        print("=" * 80)
        print(df_results)
        print("\n" + "=" * 80)
        print("📈 SUMMARY STATISTICS")
        print("=" * 80)
        print(df_results.describe())
        print("\n")


def main():  # ✅ NOT async def
    """Main function"""
    evaluator = RagasEvaluator(dataset_path=DATASET_PATH)
    
    try:
        # Setup Gemini
        evaluator.setup_gemini()
        
        # Load dataset
        evaluator.load_dataset()
        
        # Run evaluation - ✅ No await
        results = evaluator.evaluate()
        
        # Save results
        df_results = evaluator.save_results(results)
        
        # Display results
        evaluator.display_results(df_results)
        
        print("✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main() 
