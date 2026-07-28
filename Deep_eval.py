import pytest
from deepeval import assert_test, evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric
)
from deepeval.test_case import LLMTestCase

# ==========================================
# 1. DEFINE YOUR TEST DATA (OR RAG OUTPUTS)
# ==========================================

sample_user_input = "What is the return policy for defective electronics?"
retrieved_context = [
    "All electronic items carry a standard 30-day warranty.",
    "Defective electronics can be returned within 30 days for a full refund or exchange.",
    "Non-defective open-box items incur a 15% restocking fee."
]
llm_generated_output = "You can return defective electronics within 30 days for a full refund or exchange."
ground_truth_expected = "Defective electronics are eligible for full refund or replacement within 30 days."

# Construct the LLM Test Case
test_case = LLMTestCase(
    input=sample_user_input,
    actual_output=llm_generated_output,
    expected_output=ground_truth_expected,
    retrieval_context=retrieved_context
)

# ==========================================
# 2. CONFIGURE METRICS & THRESHOLDS
# ==========================================

# A. Faithfulness: Checks if the actual output contains hallucinated claims NOT in retrieval context
faithfulness = FaithfulnessMetric(threshold=0.7, model="gpt-4o")

# B. Answer Relevancy: Checks if the answer directly addresses the user query
relevancy = AnswerRelevancyMetric(threshold=0.7, model="gpt-4o")

# C. Contextual Recall: Checks if retrieved context has all info needed to formulate expected output
recall = ContextualRecallMetric(threshold=0.7, model="gpt-4o")

# D. Contextual Precision: Checks if relevant nodes in retrieval_context are ranked higher than irrelevant ones
precision = ContextualPrecisionMetric(threshold=0.7, model="gpt-4o")

# ==========================================
# 3. RUN EVALUATIONS
# ==========================================

def test_rag_quality():
    """Pytest-style assertion test"""
    assert_test(
        test_case=test_case,
        metrics=[faithfulness, relevancy, recall, precision]
    )

if __name__ == "__main__":
    # Standalone evaluation run (generates detailed score explanations)
    results = evaluate(
        test_cases=[test_case],
        metrics=[faithfulness, relevancy, recall, precision]
    )
    
    # Inspect reasoning behind scores
    print(f"\n--- Faithfulness Score: {faithfulness.score} ---")
    print(f"Reason: {faithfulness.reason}")
    print(f"\n--- Relevancy Score: {relevancy.score} ---")
    print(f"Reason: {relevancy.reason}")