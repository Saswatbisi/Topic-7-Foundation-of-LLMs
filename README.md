# Topic 7: LLM Evaluation, Hallucination Detection & Observability

## Lesson 1: Production-Grade Evaluation Frameworks, Hallucination Detection Techniques, and Real-Time Telemetry Systems

Welcome to **Topic 7: LLM Evaluation, Hallucination Detection & Observability**. This repository contains the complete theoretical foundations, practical implementation exercises, benchmark tools, and automated verification test suites for evaluating Large Language Model (LLM) outputs, detecting hallucinations, and instrumenting real-time telemetry.

---

## 1. 120-Minute Structured Class Timeline

| Time Interval | Segment Type | Topic / Sub-activity | Pedagogy & Instructor Actions | Expected Student Output |
| :--- | :--- | :--- | :--- | :--- |
| **00:00 - 00:15** | Lecture | Deterministic & Semantic Offline Metrics | Theoretical breakdown of exact match, BLEU, ROUGE-N/L, and BERTScore math. Derive contextual embedding alignment matrices. | Mathematical understanding of lexical vs semantic metrics trade-offs. |
| **00:15 - 00:35** | Lecture | LLM-as-a-Judge & Position Bias | Mechanics of single-answer and pairwise grading. Formalization of position bias, verbosity bias, and mitigation algorithms. | Bidirectional order-swap algorithm derivation. |
| **00:35 - 00:50** | Lecture | Hallucination Taxonomy & Detection Logic | Faithfulness vs. Factuality. Derive NLI-based entailment scoring and SelfCheckGPT sampling-based consistency formulas. | Sentence-level NLI entailment scoring formula. |
| **00:50 - 01:05** | Lecture | Telemetry & Observability Architecture | Real-time observability: TTFT, TPOT, E2E Latency, OpenTelemetry trace spans, and async non-blocking ring buffers. | OpenTelemetry trace span hierarchy and latency metric definitions. |
| **01:05 - 01:45** | Lab | Building an LLM Eval & Guardrail Pipeline | Hands-on coding: Implement a complete evaluation pipeline from scratch incorporating offline metrics, NLI faithfulness, judge bias mitigation, and tracing. | Runnable `eval_engine.py` pipeline. |
| **01:45 - 02:00** | Q&A / Test | Verification & Architectural Review | Run assertion test suites on synthetic payloads (faithful vs. hallucinated runs). Validate observability metric outputs. | Verification test suite `verify_lab.py` execution. |

---

## 2. Visual Sandbox & Coding Setup

### 2.1 Terminal Commands: Environment Provisioning
```bash
# Update base system and install build tools
sudo apt-get update && sudo apt-get install -y build-essential python3-dev python3-venv git

# Create and activate virtual environment
python3 -m venv llm_eval_env
source llm_eval_env/bin/activate

# Upgrade pip, setuptools, and wheel
pip install --upgrade pip setuptools wheel

# Install core evaluation, NLP, and observability libraries
pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.38.2 datasets==2.18.0 accelerate==0.27.2
pip install nltk==3.8.1 rouge-score==0.1.2 bert-score==0.3.13
pip install openai==1.14.1 pydantic==2.6.4 numpy==1.26.4
pip install opentelemetry-api==1.23.0 opentelemetry-sdk==1.23.0 prometheus-client==0.20.0

# Download NLTK tokenizers
python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 2.2 Pre-flight Sanity Diagnostic Check
Run the pre-flight sanity diagnostic script:
```bash
python sanity_check.py
```

---

## 3. Detailed Lecture Notes & Mathematical Foundations

### 3.1 Offline Evaluation Metrics Mechanics

Deterministic evaluation relies on n-gram overlap and vector similarity between candidate output ($C$) and reference text ($R$).

```
 +------------------------------------+
 |        Generated Answer (C)        |
 +------------------------------------+
                   |
     +-------------+-------------+
     |                           |
     v                           v
 +-----------------------+   +-----------------------+
 |   Lexical Matching    |   |  Semantic Alignment   |
 | (ROUGE, BLEU, Exact)  |   |      (BERTScore)      |
 +-----------------------+   +-----------------------+
             |                           |
             v                           v
   n-gram counting / precision    Transformer Token Embeddings
     & recall calculation         Cosine Similarity Matrix
```

#### ROUGE (Recall-Oriented Understudy for Gifting Evaluation)
1. **ROUGE-N**: Measures $n$-gram recall.
$$\text{ROUGE-N} = \frac{\sum_{S \in \text{References}} \sum_{\text{gram}_n \in S} \text{Count}_{\text{match}}(\text{gram}_n)}{\sum_{S \in \text{References}} \sum_{\text{gram}_n \in S} \text{Count}(\text{gram}_n)}$$

2. **ROUGE-L**: Based on Longest Common Subsequence (LCS).
$$R_{\text{lcs}} = \frac{\text{LCS}(R, C)}{|R|}, \quad P_{\text{lcs}} = \frac{\text{LCS}(R, C)}{|C|}$$
$$\text{ROUGE-L} = \frac{(1 + \beta^2) R_{\text{lcs}} P_{\text{lcs}}}{R_{\text{lcs}} + \beta^2 P_{\text{lcs}}}$$

#### BLEU (Bilingual Evaluation Understudy)
BLEU measures modified $n$-gram precision with a brevity penalty ($\text{BP}$):
$$\text{BLEU} = \text{BP} \cdot \exp \left( \sum_{n=1}^N w_n \log p_n \right)$$
$$\text{BP} = \begin{cases} 1 & \text{if } |C| > |R| \\ \exp\left(1 - \frac{|R|}{|C|}\right) & \text{if } |C| \le |R| \end{cases}$$

#### BERTScore
BERTScore computes token-level pairwise cosine similarities using contextual embeddings:
$$R_{\text{BERT}} = \frac{1}{|R|} \sum_{x_i \in R} \max_{y_j \in C} \left( \mathbf{x}_i^\top \mathbf{y}_j \right), \quad P_{\text{BERT}} = \frac{1}{|C|} \sum_{y_j \in C} \max_{x_i \in R} \left( \mathbf{x}_i^\top \mathbf{y}_j \right)$$
$$F_{\text{BERT}} = 2 \cdot \frac{P_{\text{BERT}} \cdot R_{\text{BERT}}}{P_{\text{BERT}} + R_{\text{BERT}}}$$

---

### 3.2 LLM-as-a-Judge Paradigms & Bias Mitigation

#### Position Bias Mitigation: Swap Evaluation
To eliminate order bias in pairwise comparisons, run two symmetric evaluation passes by swapping candidate presentation order:

$$\text{Pass 1} = \text{Evaluate}(A, B), \quad \text{Pass 2} = \text{Evaluate}(B, A)$$

- **Pass 1 Input**: `[ Prompt ] --> [ Model A Response ] vs [ Model B Response ]`
- **Pass 2 Input**: `[ Prompt ] --> [ Model B Response ] vs [ Model A Response ]`

**Decision Logic**:
- If `Pass 1 == A` and `Pass 2 == A` $\implies$ Win: Model A
- If `Pass 1 == B` and `Pass 2 == B` $\implies$ Win: Model B
- If `Pass 1 != Pass 2` $\implies$ Position Bias Detected / Tie

---

### 3.3 Hallucination Detection Taxonomy

```
                     +------------------------+
                     | Hallucination Taxonomy |
                     +------------------------+
                                  |
            +---------------------+---------------------+
            |                                           |
            v                                           v
 +------------------------+                  +------------------------+
 |      Faithfulness      |                  |       Factuality       |
 | (Internal Consistency) |                  | (External Ground Truth)|
 +------------------------+                  +------------------------+
 | Deviates from provided |                  | Deviates from real-    |
 | context (RAG context)  |                  | world knowledge bases  |
 +------------------------+                  +------------------------+
```

#### NLI-based Faithfulness Detection
Natural Language Inference (NLI) maps context-answer pairs to entailment relations:
- **Premise ($P$)**: Retrieved Context ($C$)
- **Hypothesis ($H$)**: Generated Sentence ($S_i$)

$$\text{Faithfulness Score} = \frac{1}{M} \sum_{i=1}^M \mathbb{I}\Big( P(\text{Entailment} \mid C, S_i) > \tau \Big)$$

---

### 3.4 Telemetry & Observability Architecture

#### Latency Metrics Formalization
- **Time To First Token (TTFT)**: Time elapsed from initial request ($t_{\text{req}}$) to first token ($t_{\text{token}_1}$):
$$\text{TTFT} = t_{\text{token}_1} - t_{\text{req}}$$

- **Time Per Output Token (TPOT)**: Average generation time per remaining token:
$$\text{TPOT} = \frac{t_{\text{completion}} - t_{\text{token}_1}}{K - 1}$$

---

## 4. Practical Lab Code Walkthroughs

### Running Core Evaluation Pipeline
File: [`eval_engine.py`](file:///d:/Tayana/Foundation%20of%20LLMs/Topic-7/eval_engine.py)
Run via:
```bash
python eval_engine.py
```

---

## 5. Automated Verification Test Suite

### Running Verification Suite
File: [`verify_lab.py`](file:///d:/Tayana/Foundation%20of%20LLMs/Topic-7/verify_lab.py)
Run via:
```bash
python verify_lab.py
```

---

## 6. Practical Lab Grading Rubric & Deduction Matrix

### Scoring Rubric (100 Points Total)

| Category | Criteria / Deliverables | Max Points | Scoring Breakdown |
| :--- | :--- | :--- | :--- |
| **1. Environment Setup & Dependency Management** | Virtual environment configuration, environment variables, deterministic pins. | 10 | **10 pts**: Fully compliant. Keys loaded safely. <br> **-5 pts**: Hardcoded API keys. <br> **-3 pts**: Unpinned dependencies. |
| **2. Core Implementation Logic** | Dual evaluation engine (ROUGE/BLEU), LLM-as-a-judge JSON parser, NLI faithfulness pipeline. | 40 | **35-40 pts**: Fully functional offline & NLI engines. <br> **20-34 pts**: Partial implementation. <br> **0 pts**: Missing core logic. |
| **3. Code Correctness & Edge Case Handling** | API rate limit backoff, bidirectional position swap, empty context/string handling. | 20 | **18-20 pts**: Flawless edge case handling. <br> **10-17 pts**: Missing position swap or retry backoff. <br> **0 pts**: Unhandled exceptions on basic inputs. |
| **4. Verification Testing & Validation Suite** | Pytest/Unittest suite covering metric computation, 15+ benchmark synthetic dataset. | 15 | **14-15 pts**: Comprehensive assertion test suite. <br> **8-13 pts**: Dataset < 10 samples. <br> **0 pts**: No test suite. |
| **5. Code Quality & Observability Instrumentation** | OpenTelemetry trace span instrumentation, type annotations, modular PEP 8 architecture, README guide. | 15 | **14-15 pts**: Complete OTel trace trees & metrics. PEP 8 clean. <br> **8-13 pts**: Tracing enabled but missing custom span attributes. <br> **0 pts**: No tracing. |

### Detailed Deduction Matrix

| Violation / Imperfection | Point Deduction | Category |
| :--- | :--- | :--- |
| Hardcoded API Key anywhere in the repository | -5 pts | Category 1 |
| Missing bidirectional swap in LLM-as-a-Judge evaluation | -5 pts | Category 3 |
| Missing Pydantic / Schema enforcement on Judge JSON output | -5 pts | Category 2 |
| Blocking synchronous network calls inside asynchronous telemetry loop | -4 pts | Category 5 |
| Missing exponential backoff on API rate limit retry blocks | -3 pts | Category 3 |
| Unhandled exception on empty retrieved context list | -3 pts | Category 3 |
| Metric scores missing from OpenTelemetry span attributes | -2 pts | Category 5 |
| Non-pep8 compliant variable naming / missing type annotations | -2 pts | Category 5 |
