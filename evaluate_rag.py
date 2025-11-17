from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from datasets import Dataset
from generator_pipeline import generate_rag_response
from retrieval_pipeline import hybrid_search, semantic_search, keyword_search
import pandas as pd


TEST_QUESTIONS = [
    {
        "question": "What is the key difference between the Transformer model and dominant sequence transduction models, and what were the experimental results on the WMT 2014 English-to-German task?",
        "expected_answer": "The Transformer is based solely on **attention mechanisms**, dispensing with recurrence and convolutions entirely, unlike dominant models which are based on complex recurrent or convolutional neural networks[cite: 17]. [cite_start]On the WMT 2014 English-to-German translation task, the Transformer achieved **28.4 BLEU**, improving over existing best results, including ensembles, by over 2 BLEU[cite: 19].",
        "ground_truth_contexts": [
        "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. . . [cite_start]We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. [cite: 15, 17]",
        "Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. [cite: 19]",
        "On the WMT 2014 English-to-German translation task, the big transformer model (Transformer (big) in Table 2) outperforms the best previously reported models (including ensembles) by more than 2.0 BLEU, establishing a new state-of-the-art BLEU score of 28.4. [cite: 228]"
        ]
    },
    {
        "question": "What are the three different ways Multi-Head Attention is used in the Transformer model?",
        "expected_answer": "Multi-Head Attention is used in three different ways in the Transformer[cite: 135]:\n* **Encoder-Decoder Attention** layers: Queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. [cite_start]This allows every position in the decoder to attend over all positions in the input sequence[cite: 136, 137].\n* **Encoder Self-Attention** layers: All of the keys, values, and queries come from the output of the previous layer in the encoder. [cite_start]Each position can attend to all positions in the previous layer of the encoder[cite: 139, 140].\n* **Decoder Self-Attention** layers: Each position in the decoder can attend to all positions in the decoder up to and including that position. [cite_start]This sub-layer is modified (masked) to prevent positions from attending to subsequent positions to preserve the auto-regressive property[cite: 141, 142].",
        "ground_truth_contexts": [
            "The Transformer uses multi-head attention in three different ways: [cite: 135]",
            "In \"encoder-decoder attention\" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. [cite: 136]",
            "This allows every position in the decoder to attend over all positions in the input sequence. [cite: 137]",
            "The encoder contains self-attention layers. [cite_start]In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder. [cite: 139]",
            "Each position in the encoder can attend to all positions in the previous layer of the encoder. [cite: 140]",
            "Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position. [cite: 141]",
            "We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. [cite: 142]",
            "We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. [cite: 86]"
        ]
    },
    {
        "question": "What method is used in the Transformer to inject information about the sequence order, and why was this particular method chosen over a learned version?",
        "expected_answer": "The Transformer uses **positional encodings** added to the input embeddings at the bottom of the encoder and decoder stacks to inject information about the relative or absolute position of the tokens in the sequence, since the model contains no recurrence or convolution[cite: 162, 163]. [cite_start]The positional encodings are implemented using **sine and cosine functions of different frequencies**[cite: 166]. [cite_start]This sinusoidal version was chosen because the authors hypothesized it may allow the model to **extrapolate to sequence lengths longer** than the ones encountered during training [cite: 173][cite_start], even though experimenting with learned positional embeddings produced nearly identical results[cite: 172].",
        "ground_truth_contexts": [
            "Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence. [cite: 162]",
            "To this end, we add \"positional encodings\" to the input embeddings at the bottoms of the encoder and decoder stacks. [cite: 163]",
            "In this work, we use sine and cosine functions of different frequencies: [cite: 166]",
            "We also experimented with using learned positional embeddings [9] instead, and found that the two versions produced nearly identical results (see Table 3 row (E)). [cite: 172]",
            "We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training. [cite: 173]"
        ]
    }
]


def run_evaluation():
    print("Starting RAG Evaluation...\n")

    # Containers for dataset - FIXED: Added reference_contexts initialization
    questions, answers, contexts, ground_truths, reference_contexts = [], [], [], [], []

    # Process each test question
    for test in TEST_QUESTIONS:
        question = test["question"]
        expected_answer = test["expected_answer"]
        gt_contexts = test["ground_truth_contexts"]

        print(f"Processing: {question}")

        # Generate RAG answer
        answer = generate_rag_response(
            query=question,
            search_type="hybrid",  # or "hybrid"/"keyword" depending on your setup
            top_k=5
        )

        # Retrieve supporting contexts
        retrieved = hybrid_search(question, top_k=5)
        retrieved_contexts = [r["_source"]["content"] for r in retrieved]

        # Store results
        questions.append(question)
        answers.append(answer)
        contexts.append(retrieved_contexts)      # retrieved contexts
        ground_truths.append(expected_answer)    # only the string answer
        reference_contexts.append(gt_contexts)   # list of ground-truth contexts

    # Create RAGAS-compatible dataset
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
        "reference_contexts": reference_contexts  # added for context metrics
    })

    # Evaluate using RAGAS metrics
    print("\nEvaluating...")
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision]
    )

    results_df = results.to_pandas()

    # Compute average scores
    faithfulness_score = results_df["faithfulness"].mean()
    answer_relevancy_score = results_df["answer_relevancy"].mean()
    context_recall_score = results_df["context_recall"].mean()
    context_precision_score = results_df["context_precision"].mean()

    avg_score = (
        faithfulness_score +
        answer_relevancy_score +
        context_recall_score +
        context_precision_score
    ) / 4

    # Display results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"\nFaithfulness Score:     {faithfulness_score:.3f}")
    print(f"Answer Relevancy:       {answer_relevancy_score:.3f}")
    print(f"Context Recall:         {context_recall_score:.3f}")
    print(f"Context Precision:      {context_precision_score:.3f}")
    print(f"\nAverage Score:          {avg_score:.3f}")

    # Interpretation
    print("\n" + "="*50)
    print("INTERPRETATION")
    print("="*50)
    if avg_score >= 0.8:
        print("✓ Excellent - System is performing very well")
    elif avg_score >= 0.6:
        print("✓ Good - System is performing adequately")
    elif avg_score >= 0.4:
        print("⚠ Fair - There's room for improvement")
    else:
        print("✗ Needs Work - Significant optimization needed")

    # Optional: show per-question metrics
    print("\nDetailed per-question metrics:\n")
    print(results_df.round(3))

    # Save results
    results_df.to_csv("evaluation_results.csv", index=False)
    print("\nDetailed results saved to: evaluation_results.csv")

    return results_df


if __name__ == "__main__":
    run_evaluation()