import time
from collections import Counter
import pandas as pd
from data_loader import load_dataset, build_vocabulary, UNK_TOKEN
from hmm import FirstOrderHMM
from hmm_second_order import SecondOrderHMM

def calculate_metrics(gold_labels, pred_labels, sentences, vocab):
    """
    Calculate comprehensive evaluation metrics:
    - Token Accuracy (Overall, Known Words, Unknown Words)
    - Sentence-Level (Sequence) Accuracy
    - Top Confusion Errors
    """
    total_tokens = 0
    correct_tokens = 0
    
    known_total = 0
    known_correct = 0
    
    unk_total = 0
    unk_correct = 0
    
    total_sentences = len(gold_labels)
    correct_sentences = 0
    
    error_pairs = Counter()
    
    for sent, gold_seq, pred_seq in zip(sentences, gold_labels, pred_labels):
        sent_correct = True
        for word, gold, pred in zip(sent, gold_seq, pred_seq):
            total_tokens += 1
            is_known = word in vocab and word != UNK_TOKEN
            
            if gold == pred:
                correct_tokens += 1
                if is_known:
                    known_total += 1
                    known_correct += 1
                else:
                    unk_total += 1
                    unk_correct += 1
            else:
                sent_correct = False
                error_pairs[(gold, pred)] += 1
                if is_known:
                    known_total += 1
                else:
                    unk_total += 1
                    
        if sent_correct:
            correct_sentences += 1
            
    return {
        "total_tokens": total_tokens,
        "token_accuracy": correct_tokens / total_tokens if total_tokens > 0 else 0.0,
        "known_accuracy": known_correct / known_total if known_total > 0 else 0.0,
        "unknown_accuracy": unk_correct / unk_total if unk_total > 0 else 0.0,
        "known_count": known_total,
        "unknown_count": unk_total,
        "sentence_accuracy": correct_sentences / total_sentences if total_sentences > 0 else 0.0,
        "total_sentences": total_sentences,
        "correct_sentences": correct_sentences,
        "top_errors": error_pairs.most_common(5)
    }

def print_separator(title=""):
    print("\n" + "=" * 70)
    if title:
        print(f" {title.upper()} ".center(70, "="))
        print("=" * 70)

def evaluate_learning_curve(train_sentences, train_labels, dev_sentences, dev_labels, vocab, fractions=[0.1, 0.25, 0.5, 1.0]):
    """
    Evaluate how First-Order and Second-Order HMM scale with varying amounts of training data.
    """
    print_separator("Learning Curve (Data Efficiency Benchmark)")
    results = []
    
    for frac in fractions:
        n_samples = int(len(train_sentences) * frac)
        sub_train_sents = train_sentences[:n_samples]
        sub_train_labels = train_labels[:n_samples]
        
        # Build sub-vocabulary for fair comparison
        sub_vocab, _ = build_vocabulary(sub_train_sents, min_freq=2)
        
        # 1. Train & evaluate First-Order HMM
        h1 = FirstOrderHMM()
        h1.train(sub_train_sents, sub_train_labels, sub_vocab)
        h1_preds = h1.predict(dev_sentences)
        m1 = calculate_metrics(dev_labels, h1_preds, dev_sentences, sub_vocab)
        
        # 2. Train & evaluate Second-Order HMM
        h2 = SecondOrderHMM()
        h2.train(sub_train_sents, sub_train_labels, sub_vocab)
        h2_preds = h2.predict(dev_sentences)
        m2 = calculate_metrics(dev_labels, h2_preds, dev_sentences, sub_vocab)
        
        results.append({
            "Train Size": f"{frac*100:.0f}% ({n_samples:,})",
            "1st-Order Token Acc": f"{m1['token_accuracy']*100:.2f}%",
            "2nd-Order Token Acc": f"{m2['token_accuracy']*100:.2f}%",
            "1st-Order Sent Acc": f"{m1['sentence_accuracy']*100:.2f}%",
            "2nd-Order Sent Acc": f"{m2['sentence_accuracy']*100:.2f}%"
        })
        print(f"  Processed {frac*100:.0f}% data: 1st-Order Token Acc={m1['token_accuracy']*100:.2f}% | 2nd-Order Token Acc={m2['token_accuracy']*100:.2f}%")
        
    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))

def main():
    print_separator("Loading Dataset")
    (train_sentences, train_labels), (dev_sentences, dev_labels) = load_dataset()
    
    print(f"Loaded {len(train_sentences):,} training sentences.")
    print(f"Loaded {len(dev_sentences):,} development/evaluation sentences.")
    
    vocab, word_counts = build_vocabulary(train_sentences, min_freq=2)
    print(f"Vocabulary size: {len(vocab):,} unique words (min_freq >= 2 + UNK).")
    
    # -------------------------------------------------------------
    # 1. First-Order HMM Evaluation
    # -------------------------------------------------------------
    print_separator("Training First-Order HMM")
    t0 = time.time()
    hmm1 = FirstOrderHMM(emission_k=1e-3, transition_k=1e-3)
    hmm1.train(train_sentences, train_labels, vocab)
    hmm1_train_time = time.time() - t0
    print(f"First-Order HMM trained in {hmm1_train_time:.3f} seconds.")
    
    print("\nRunning First-Order Viterbi Decoding on Dev Set...")
    t0 = time.time()
    hmm1_preds = hmm1.predict(dev_sentences)
    hmm1_eval_time = time.time() - t0
    m1 = calculate_metrics(dev_labels, hmm1_preds, dev_sentences, vocab)
    print(f"First-Order HMM evaluated in {hmm1_eval_time:.3f} seconds ({len(dev_sentences)/hmm1_eval_time:.1f} sent/sec).")
    
    # -------------------------------------------------------------
    # 2. Second-Order HMM Evaluation
    # -------------------------------------------------------------
    print_separator("Training Second-Order HMM")
    t0 = time.time()
    hmm2 = SecondOrderHMM(emission_k=1e-3)
    hmm2.train(train_sentences, train_labels, vocab)
    hmm2_train_time = time.time() - t0
    print(f"Second-Order HMM trained in {hmm2_train_time:.3f} seconds.")
    print(f"Calculated Deleted Interpolation Lambdas: (lambda1={hmm2.lambdas[0]:.4f}, lambda2={hmm2.lambdas[1]:.4f}, lambda3={hmm2.lambdas[2]:.4f})")
    
    print("\nRunning Second-Order Viterbi Decoding on Dev Set...")
    t0 = time.time()
    hmm2_preds = hmm2.predict(dev_sentences)
    hmm2_eval_time = time.time() - t0
    m2 = calculate_metrics(dev_labels, hmm2_preds, dev_sentences, vocab)
    print(f"Second-Order HMM evaluated in {hmm2_eval_time:.3f} seconds ({len(dev_sentences)/hmm2_eval_time:.1f} sent/sec).")

    # -------------------------------------------------------------
    # 3. Side-by-Side Comparison
    # -------------------------------------------------------------
    print_separator("Final Comparative Results")
    summary = [
        {"Metric": "Overall Token Accuracy", "First-Order HMM": f"{m1['token_accuracy']*100:.2f}%", "Second-Order HMM": f"{m2['token_accuracy']*100:.2f}%"},
        {"Metric": "Known Words Accuracy", "First-Order HMM": f"{m1['known_accuracy']*100:.2f}%", "Second-Order HMM": f"{m2['known_accuracy']*100:.2f}%"},
        {"Metric": "Unknown Words Accuracy", "First-Order HMM": f"{m1['unknown_accuracy']*100:.2f}%", "Second-Order HMM": f"{m2['unknown_accuracy']*100:.2f}%"},
        {"Metric": "Sentence Accuracy (Exact)", "First-Order HMM": f"{m1['sentence_accuracy']*100:.2f}%", "Second-Order HMM": f"{m2['sentence_accuracy']*100:.2f}%"},
        {"Metric": "Training Time", "First-Order HMM": f"{hmm1_train_time:.3f} s", "Second-Order HMM": f"{hmm2_train_time:.3f} s"},
        {"Metric": "Inference Time (5.5k sents)", "First-Order HMM": f"{hmm1_eval_time:.3f} s", "Second-Order HMM": f"{hmm2_eval_time:.3f} s"},
        {"Metric": "Inference Speed", "First-Order HMM": f"{len(dev_sentences)/hmm1_eval_time:.1f} sent/s", "Second-Order HMM": f"{len(dev_sentences)/hmm2_eval_time:.1f} sent/s"}
    ]
    df_summary = pd.DataFrame(summary)
    print(df_summary.to_string(index=False))
    
    # -------------------------------------------------------------
    # 4. Error Analysis
    # -------------------------------------------------------------
    print_separator("Top 5 Most Common Tagging Errors")
    print("\nFirst-Order HMM Top Errors [True Tag -> Predicted Tag]:")
    for (gold, pred), cnt in m1["top_errors"]:
        print(f"  {gold:>6} -> {pred:<6} : {cnt:>5} occurrences")
        
    print("\nSecond-Order HMM Top Errors [True Tag -> Predicted Tag]:")
    for (gold, pred), cnt in m2["top_errors"]:
        print(f"  {gold:>6} -> {pred:<6} : {cnt:>5} occurrences")

    # -------------------------------------------------------------
    # 5. Data Efficiency / Learning Curve
    # -------------------------------------------------------------
    evaluate_learning_curve(train_sentences, train_labels, dev_sentences, dev_labels, vocab)

if __name__ == "__main__":
    main()
