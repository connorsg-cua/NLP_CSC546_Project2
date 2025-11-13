import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import re

# Try to import ROUGE scorer, fallback if not available
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    try:
        # Alternative: use datasets library's ROUGE metric
        from datasets import load_metric
        rouge_metric = load_metric("rouge")
        ROUGE_AVAILABLE = True
        ROUGE_TYPE = "datasets"
    except ImportError:
        ROUGE_AVAILABLE = False
        ROUGE_TYPE = None
        print("Warning: ROUGE scoring library not available. Install 'rouge-score' or 'datasets' package.")

def evaluate_ner(predictions, ground_truth):
    """
    Placeholder function to evaluate NER performance.
    Assumes predictions and ground_truth are lists of (word, entity_type) tuples.
    """
    y_true = [entity for _, entity in ground_truth]
    y_pred = [entity for _, entity in predictions]

    # Align lengths if necessary (simple truncation for demonstration)
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    if not y_true or not y_pred:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}

    # Calculate metrics for each entity type
    metrics = precision_recall_fscore_support(y_true, y_pred, average=None, labels=list(set(y_true + y_pred)))
    
    # Aggregate for overall metrics (e.g., macro average)
    overall_metrics = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)

    results = {
        "overall_precision": overall_metrics[0],
        "overall_recall": overall_metrics[1],
        "overall_f1": overall_metrics[2],
        "overall_support": overall_metrics[3],
        "per_entity_metrics": {}
    }

    labels = list(set(y_true + y_pred))
    for i, label in enumerate(labels):
        results["per_entity_metrics"][label] = {
            "precision": metrics[0][i],
            "recall": metrics[1][i],
            "f1": metrics[2][i],
            "support": metrics[3][i]
        }
    
    return results

def evaluate_summarization(predictions, ground_truth):
    """
    Evaluate summarization performance using ROUGE scores.
    
    Args:
        predictions: List of predicted summary strings or single string
        ground_truth: List of reference summary strings or single string
    
    Returns:
        Dictionary with ROUGE-1, ROUGE-2, and ROUGE-L scores
    """
    if not ROUGE_AVAILABLE:
        return {
            "rouge1_f1": 0.0, 
            "rouge1_precision": 0.0,
            "rouge1_recall": 0.0,
            "rouge2_f1": 0.0,
            "rouge2_precision": 0.0,
            "rouge2_recall": 0.0,
            "rougeL_f1": 0.0,
            "rougeL_precision": 0.0,
            "rougeL_recall": 0.0,
            "note": "ROUGE library not available. Install 'rouge-score' or 'datasets' package."
        }
    
    # Handle single strings
    if isinstance(predictions, str):
        predictions = [predictions]
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    
    # Ensure same length
    min_len = min(len(predictions), len(ground_truth))
    predictions = predictions[:min_len]
    ground_truth = ground_truth[:min_len]
    
    if ROUGE_TYPE == "datasets":
        # Use datasets library
        results = rouge_metric.compute(
            predictions=predictions,
            references=ground_truth
        )
        return {
            "rouge1_f1": results["rouge1"].mid.fmeasure,
            "rouge1_precision": results["rouge1"].mid.precision,
            "rouge1_recall": results["rouge1"].mid.recall,
            "rouge2_f1": results["rouge2"].mid.fmeasure,
            "rouge2_precision": results["rouge2"].mid.precision,
            "rouge2_recall": results["rouge2"].mid.recall,
            "rougeL_f1": results["rougeL"].mid.fmeasure,
            "rougeL_precision": results["rougeL"].mid.precision,
            "rougeL_recall": results["rougeL"].mid.recall,
        }
    else:
        # Use rouge_score library
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Calculate average scores across all pairs
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for pred, ref in zip(predictions, ground_truth):
            scores = scorer.score(ref, pred)
            rouge1_scores.append(scores['rouge1'])
            rouge2_scores.append(scores['rouge2'])
            rougeL_scores.append(scores['rougeL'])
        
        # Average the scores
        def avg_scores(score_list):
            return {
                "f1": sum(s.fmeasure for s in score_list) / len(score_list),
                "precision": sum(s.precision for s in score_list) / len(score_list),
                "recall": sum(s.recall for s in score_list) / len(score_list)
            }
        
        rouge1_avg = avg_scores(rouge1_scores)
        rouge2_avg = avg_scores(rouge2_scores)
        rougeL_avg = avg_scores(rougeL_scores)
        
        return {
            "rouge1_f1": rouge1_avg["f1"],
            "rouge1_precision": rouge1_avg["precision"],
            "rouge1_recall": rouge1_avg["recall"],
            "rouge2_f1": rouge2_avg["f1"],
            "rouge2_precision": rouge2_avg["precision"],
            "rouge2_recall": rouge2_avg["recall"],
            "rougeL_f1": rougeL_avg["f1"],
            "rougeL_precision": rougeL_avg["precision"],
            "rougeL_recall": rougeL_avg["recall"],
        }

def create_ground_truth_summaries(texts):
    """
    Create reference summaries for evaluation.
    In a real scenario, these would be human-written summaries.
    For now, we create simple extractive summaries as ground truth.
    
    Args:
        texts: List of input texts
    
    Returns:
        List of reference summaries
    """
    ground_truth = []
    for text in texts:
        # Simple extractive summary: first sentence + key phrases
        sentences = re.split(r'[.!?]+', text)
        if sentences:
            # Take first sentence as base
            summary = sentences[0].strip()
            # Add key information if available
            if len(sentences) > 1:
                # Look for important keywords
                important_words = ['completed', 'fixed', 'implemented', 'resolved', 'created', 'updated']
                for sent in sentences[1:3]:  # Check next 2 sentences
                    if any(word in sent.lower() for word in important_words):
                        summary += " " + sent.strip()
                        break
            ground_truth.append(summary[:200])  # Limit length
        else:
            ground_truth.append(text[:200])
    return ground_truth

def run_evaluation_on_data(predictions, ground_truth, task_type="summarization"):
    """
    Run evaluation on a dataset of predictions and ground truth.
    
    Args:
        predictions: List of predictions
        ground_truth: List of ground truth labels/summaries
        task_type: "summarization" or "ner"
    
    Returns:
        Dictionary with evaluation metrics
    """
    if task_type == "summarization":
        return evaluate_summarization(predictions, ground_truth)
    elif task_type == "ner":
        return evaluate_ner(predictions, ground_truth)
    else:
        raise ValueError(f"Unknown task type: {task_type}")

def generate_evaluation_report(ner_results=None, summarization_results=None):
    """
    Generate a formatted evaluation report.
    
    Args:
        ner_results: Dictionary with NER evaluation results
        summarization_results: Dictionary with summarization evaluation results
    
    Returns:
        Formatted string report
    """
    report = "=" * 60 + "\n"
    report += "EVALUATION REPORT\n"
    report += "=" * 60 + "\n\n"
    
    if ner_results:
        report += "NER EVALUATION RESULTS\n"
        report += "-" * 60 + "\n"
        report += f"Overall Precision: {ner_results.get('overall_precision', 0):.4f}\n"
        report += f"Overall Recall: {ner_results.get('overall_recall', 0):.4f}\n"
        report += f"Overall F1-Score: {ner_results.get('overall_f1', 0):.4f}\n"
        report += f"Overall Support: {ner_results.get('overall_support', 0)}\n\n"
        
        if 'per_entity_metrics' in ner_results:
            report += "Per-Entity Metrics:\n"
            for entity, metrics in ner_results['per_entity_metrics'].items():
                report += f"  {entity}:\n"
                report += f"    Precision: {metrics['precision']:.4f}\n"
                report += f"    Recall: {metrics['recall']:.4f}\n"
                report += f"    F1: {metrics['f1']:.4f}\n"
                report += f"    Support: {metrics['support']}\n"
        report += "\n"
    
    if summarization_results:
        report += "SUMMARIZATION EVALUATION RESULTS (ROUGE)\n"
        report += "-" * 60 + "\n"
        report += "ROUGE-1:\n"
        report += f"  F1: {summarization_results.get('rouge1_f1', 0):.4f}\n"
        report += f"  Precision: {summarization_results.get('rouge1_precision', 0):.4f}\n"
        report += f"  Recall: {summarization_results.get('rouge1_recall', 0):.4f}\n\n"
        
        report += "ROUGE-2:\n"
        report += f"  F1: {summarization_results.get('rouge2_f1', 0):.4f}\n"
        report += f"  Precision: {summarization_results.get('rouge2_precision', 0):.4f}\n"
        report += f"  Recall: {summarization_results.get('rouge2_recall', 0):.4f}\n\n"
        
        report += "ROUGE-L:\n"
        report += f"  F1: {summarization_results.get('rougeL_f1', 0):.4f}\n"
        report += f"  Precision: {summarization_results.get('rougeL_precision', 0):.4f}\n"
        report += f"  Recall: {summarization_results.get('rougeL_recall', 0):.4f}\n\n"
        
        if 'note' in summarization_results:
            report += f"Note: {summarization_results['note']}\n"
    
    report += "=" * 60 + "\n"
    return report

if __name__ == "__main__":
    print("--- NER Evaluation Example ---")
    # Example usage for NER
    ner_ground_truth = [
        ("John", "PERSON"), ("works", "O"), ("at", "O"), ("Google", "ORG"), ("in", "O"), ("Python", "SKILL")
    ]
    ner_predictions = [
        ("John", "PERSON"), ("works", "O"), ("at", "O"), ("Google", "ORG"), ("in", "O"), ("Python", "SKILL")
    ]
    ner_results = evaluate_ner(ner_predictions, ner_ground_truth)
    print("NER Results:", ner_results)

    ner_predictions_bad = [
        ("John", "ORG"), ("works", "O"), ("at", "O"), ("Google", "PERSON"), ("in", "O"), ("Python", "O")
    ]
    ner_results_bad = evaluate_ner(ner_predictions_bad, ner_ground_truth)
    print("\nNER Results (Bad Predictions):", ner_results_bad)

    print("\n--- Summarization Evaluation Example ---")
    # Example usage for Summarization
    summary_ground_truth = "This is a short summary of the document."
    summary_prediction = "This is a summary of the document."
    summary_results = evaluate_summarization(summary_prediction, summary_ground_truth)
    print("Summarization Results:", summary_results)
    
    print("\n--- Evaluation Report Example ---")
    report = generate_evaluation_report(ner_results=ner_results, summarization_results=summary_results)
    print(report)
