# First-Order vs. Second-Order HMM for POS Tagging

An implementation from scratch of **First-Order** and **Second-Order (Trigram) Hidden Markov Models (HMM)** in Python for Part-of-Speech (POS) tagging.

---

## 📌 Project Overview
This project compares a standard First-Order HMM with a Second-Order (Trigram) HMM on the Wall Street Journal POS tagging benchmark dataset downloaded via KaggleHub (`pranav13300/annotated-dataset-for-pos-tagging`).

### Key Features
- **Pure Scratch Implementation**: No high-level NLP libraries (e.g. NLTK, spaCy, or HuggingFace) are used for model training or inference.
- **Deleted Interpolation**: Second-order transition probabilities use Jelinek-Mercer / Brants TnT deleted interpolation to estimate optimal linear weights ($\lambda_1, \lambda_2, \lambda_3$).
- **Vectorized Viterbi Decoding**: High-performance NumPy tensor operations for both 2D (first-order) and 3D (second-order) Viterbi dynamic programming.
- **Comprehensive Evaluation**: Metrics include overall token accuracy, known vs. unknown word accuracy, sentence-level exact match, top error confusion, and learning curve data efficiency.

---

## 📐 Mathematical Formulation

### 1. First-Order HMM
For a word sequence $W = (w_1, \dots, w_n)$ and tag sequence $T = (t_1, \dots, t_n)$:
$$P(W, T) = P(t_1 \mid \text{START}) \prod_{i=2}^n P(t_i \mid t_{i-1}) P(\text{STOP} \mid t_n) \prod_{i=1}^n P(w_i \mid t_i)$$

- **Transition Probability**: Smoothed using Laplace Add-$k$:
  $$P(t_i \mid t_{i-1}) = \frac{C(t_{i-1}, t_i) + k}{C(t_{i-1}) + k \cdot (|S| + 1)}$$
- **Emission Probability**:
  $$P(w_i \mid t_i) = \frac{C(t_i, w_i) + k}{C(t_i) + k \cdot |V|}$$

### 2. Second-Order HMM
Transitions condition on the previous two tags:
$$P(W, T) = P(t_1 \mid \text{START}, \text{START}) P(t_2 \mid \text{START}, t_1) \prod_{i=3}^n P(t_i \mid t_{i-2}, t_{i-1}) P(\text{STOP} \mid t_{n-1}, t_n) \prod_{i=1}^n P(w_i \mid t_i)$$

- **Linear Interpolation Smoothing**:
  $$P(t_i \mid t_{i-2}, t_{i-1}) = \lambda_3 P_{\text{MLE}}(t_i \mid t_{i-2}, t_{i-1}) + \lambda_2 P_{\text{MLE}}(t_i \mid t_{i-1}) + \lambda_1 P_{\text{MLE}}(t_i)$$
  where $\lambda_1 + \lambda_2 + \lambda_3 = 1$.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- `uv` (Fast Python package installer)

### Setup & Installation
1. Create a virtual environment using `uv`:
   ```bash
   python -m uv venv
   ```

2. Activate the virtual environment:
   - **Windows (PowerShell)**:
     ```powershell
     .\.venv\Scripts\activate
     ```
   - **Linux / macOS**:
     ```bash
     source .venv/bin/activate
     ```

3. Install dependencies:
   ```bash
   python -m uv pip install -r requirements.txt --python .venv\Scripts\python.exe
   ```

---

## 📊 Running the Evaluation

Execute the evaluation pipeline across all benchmarks:
```bash
python evaluate.py
```

---

## 📈 Experimental Results

*Evaluated on 5,527 validation sentences (131,768 tokens) from the annotated WSJ dataset.*

### 1. Overall Performance Comparison

| Metric | First-Order HMM | Second-Order HMM | Difference |
| :--- | :---: | :---: | :---: |
| **Overall Token Accuracy** | **94.81%** | **95.34%** | `+0.53%` |
| **Known Words Accuracy** | **96.65%** | **96.97%** | `+0.32%` |
| **Unknown Words Accuracy (<UNK>)** | **58.05%** | **62.71%** | `+4.66%` |
| **Sentence-Level Exact Match** | **35.37%** | **38.94%** | `+3.57%` |
| **Training Time** | **0.695 s** | **1.023 s** | — |
| **Inference Speed** | **4,108 sent/s** | **262 sent/s** | — |

---

### 2. Learning Curve (Data Efficiency)

| Training Data Fraction | Samples | 1st-Order Token Acc | 2nd-Order Token Acc | 1st-Order Sent Acc | 2nd-Order Sent Acc |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **10%** | 3,821 | 90.25% | 91.01% | 16.92% | 19.18% |
| **25%** | 9,554 | 92.62% | 93.21% | 24.06% | 27.39% |
| **50%** | 19,109 | 93.94% | 94.45% | 30.22% | 33.53% |
| **100%** | 38,218 | 94.81% | 95.34% | 35.37% | 38.94% |

---

### 3. Top Common Confusion Errors
The top errors across both models occur on ambiguous noun/adjective/verb boundaries:
1. `NN -> JJ` (Noun misclassified as Adjective)
2. `NN -> NNP` (Common Noun misclassified as Proper Noun)
3. `VBD -> VBN` (Past Tense misclassified as Past Participle)
4. `NNS -> NN` (Plural Noun misclassified as Singular Noun)
5. `NNP -> JJ` (Proper Noun misclassified as Adjective)

The Second-Order HMM achieves significantly fewer errors in contextual ambiguities like `VBD -> VBN` and `NNS -> NN` due to capturing wider two-tag history.

---

## 📂 Project Structure
```
├── data_loader.py       # Kaggle dataset download, ingestion, and vocabulary preprocessing
├── hmm.py               # First-Order HMM implementation from scratch
├── hmm_second_order.py  # Second-Order HMM implementation from scratch
├── evaluate.py          # Comprehensive evaluation & benchmarking suite
├── requirements.txt     # Dependency definitions
├── .gitignore           # Git ignore configuration
└── README.md            # Documentation and benchmark report
```
