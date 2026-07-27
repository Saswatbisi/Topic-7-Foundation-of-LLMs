"""Production-Grade Evaluation Frameworks, Hallucination Detection Techniques, and Real-Time Telemetry Systems."""

import os
import time
import json
import math
import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict

import numpy as np
import torch

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError:
    AutoTokenizer = None
    AutoModelForSequenceClassification = None

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
except ImportError:
    sentence_bleu = None
    SmoothingFunction = None

try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None

try:
    import openai
except ImportError:
    openai = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EvalEngine")

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)
    span_processor = SimpleSpanProcessor(ConsoleSpanExporter())
    trace.get_tracer_provider().add_span_processor(span_processor)
except ImportError:
    class DummySpanContext:
        def __init__(self):
            self.trace_id = 0x1234567890abcdef

    class DummySpan:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
        def get_span_context(self):
            return DummySpanContext()
        def set_attribute(self, key, value):
            pass

    class DummyTracer:
        def start_as_current_span(self, name):
            return DummySpan()

    tracer = DummyTracer()


# =====================================================================
# Module 1: Deterministic Offline Metrics Evaluator
# =====================================================================
class DeterministicMetricsEvaluator:
    """Computes exact match, BLEU, ROUGE, and surface similarity scores."""

    def __init__(self) -> None:
        if rouge_scorer:
            self.rouge_evaluator = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
            )
        else:
            self.rouge_evaluator = None
            
        if SmoothingFunction:
            self.smoothing = SmoothingFunction().method1
        else:
            self.smoothing = None

    def evaluate_exact_match(self, prediction: str, reference: str) -> float:
        """Calculates exact string match (1.0 or 0.0) after standardization."""
        if not prediction or not reference:
            return 0.0
        p_clean = prediction.strip().lower()
        r_clean = reference.strip().lower()
        return 1.0 if p_clean == r_clean else 0.0

    def evaluate_bleu(self, prediction: str, reference: str) -> float:
        """Calculates sentence BLEU score using smoothing."""
        if not prediction or not reference:
            return 0.0
        if not sentence_bleu or not self.smoothing:
            # Fallback n-gram overlap BLEU approximation
            p_tokens = prediction.strip().lower().split()
            r_tokens = reference.strip().lower().split()
            if not p_tokens or not r_tokens:
                return 0.0
            overlap = sum(1 for t in p_tokens if t in r_tokens)
            return float(overlap / len(p_tokens))

        ref_tokens = [reference.strip().lower().split()]
        pred_tokens = prediction.strip().lower().split()
        if not pred_tokens or not ref_tokens[0]:
            return 0.0

        return float(sentence_bleu(
            ref_tokens,
            pred_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=self.smoothing
        ))

    def evaluate_rouge(self, prediction: str, reference: str) -> Dict[str, float]:
        """Calculates standard ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L)."""
        if not prediction or not reference:
            return {"rouge1_f1": 0.0, "rouge2_f1": 0.0, "rougeL_f1": 0.0}

        if not self.rouge_evaluator:
            # Fallback simple overlap calculation
            p_words = set(prediction.lower().split())
            r_words = set(reference.lower().split())
            if not p_words or not r_words:
                return {"rouge1_f1": 0.0, "rouge2_f1": 0.0, "rougeL_f1": 0.0}
            overlap = len(p_words & r_words)
            precision = overlap / len(p_words)
            recall = overlap / len(r_words)
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            return {"rouge1_f1": f1, "rouge2_f1": f1, "rougeL_f1": f1}

        scores = self.rouge_evaluator.score(reference, prediction)
        return {
            "rouge1_f1": float(scores['rouge1'].fmeasure),
            "rouge2_f1": float(scores['rouge2'].fmeasure),
            "rougeL_f1": float(scores['rougeL'].fmeasure)
        }

    def evaluate_all(self, prediction: str, reference: str) -> Dict[str, float]:
        """Runs all deterministic metric evaluations."""
        results = {
            "exact_match": self.evaluate_exact_match(prediction, reference),
            "bleu": self.evaluate_bleu(prediction, reference)
        }
        results.update(self.evaluate_rouge(prediction, reference))
        return results


# =====================================================================
# Module 2: Pairwise LLM-as-a-Judge with Bias Mitigation
# =====================================================================
class PairwiseLLMJudge:
    """Evaluates output quality using an LLM judge, applying position-swapping to reduce bias."""

    def __init__(self, model_name: str = "gpt-4o", max_retries: int = 3) -> None:
        self.model_name = model_name
        self.max_retries = max_retries
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=api_key) if (openai and api_key and api_key != "sk-proj-YOUR_API_KEY_HERE") else None

    def _build_evaluation_prompt(self, prompt: str, resp_a: str, resp_b: str) -> str:
        return f"""You are an unbiased expert evaluator assessing two model outputs.

User Prompt: {prompt}

[Response A]:
{resp_a}

[Response B]:
{resp_b}

Evaluate which response is better based on accuracy, clarity, and adherence to the prompt.
You MUST output a JSON object in exactly this format:
{{"winner": "A", "reasoning": "..."}} or {{"winner": "B", "reasoning": "..."}} or {{"winner": "TIE", "reasoning": "..."}}
"""

    def judge_pairwise(self, prompt: str, response_1: str, response_2: str) -> Dict[str, Any]:
        """Performs two-pass evaluations with swapped positions to mitigate position bias."""
        if not response_1 and not response_2:
            return {
                "final_winner": "TIE",
                "position_bias_detected": False,
                "pass1_winner": "TIE",
                "pass2_winner": "TIE",
                "pass1_reasoning": "Empty responses.",
                "pass2_reasoning": "Empty responses."
            }

        if not self.client:
            return self._mock_judge_pairwise(response_1, response_2)

        # Pass 1: Original Order (response_1 = A, response_2 = B)
        p1_text = self._build_evaluation_prompt(prompt, response_1, response_2)
        res1 = self._call_llm_with_retry(p1_text)

        # Pass 2: Swapped Order (response_2 = A, response_1 = B)
        p2_text = self._build_evaluation_prompt(prompt, response_2, response_1)
        res2 = self._call_llm_with_retry(p2_text)

        winner_p1_raw = res1.get("winner", "TIE")
        winner_p2_raw = res2.get("winner", "TIE")

        # Map Pass 1 labels (A -> Model 1, B -> Model 2)
        winner_p1 = "1" if winner_p1_raw == "A" else ("2" if winner_p1_raw == "B" else "TIE")

        # Map Pass 2 labels (A -> Model 2, B -> Model 1)
        winner_p2 = "2" if winner_p2_raw == "A" else ("1" if winner_p2_raw == "B" else "TIE")

        # Position Bias Check Logic
        if winner_p1 == winner_p2 and winner_p1 in ["1", "2"]:
            final_winner = f"Model_{winner_p1}"
            position_bias_detected = False
        elif winner_p1 == "TIE" and winner_p2 == "TIE":
            final_winner = "TIE"
            position_bias_detected = False
        else:
            final_winner = "TIE_DISAGREEMENT"
            position_bias_detected = True

        return {
            "final_winner": final_winner,
            "position_bias_detected": position_bias_detected,
            "pass1_winner": winner_p1,
            "pass2_winner": winner_p2,
            "pass1_reasoning": res1.get("reasoning", ""),
            "pass2_reasoning": res2.get("reasoning", "")
        }

    def _call_llm_with_retry(self, prompt_text: str) -> Dict[str, str]:
        """Executes LLM call with exponential backoff and jitter."""
        delay = 1.0
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt_text}],
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                content = response.choices[0].message.content
                return json.loads(content)
            except Exception as e:
                logger.warning(f"LLM Judge call failed (Attempt {attempt+1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    return {"winner": "TIE", "reasoning": f"Execution Failure: {str(e)}"}
                time.sleep(delay)
                delay *= 2.0
        return {"winner": "TIE", "reasoning": "Execution Failure: Retries exhausted"}

    def _mock_judge_pairwise(self, response_1: str, response_2: str) -> Dict[str, Any]:
        """Deterministic mock evaluation for offline testing."""
        len1 = len(response_1 or "")
        len2 = len(response_2 or "")
        winner = "Model_1" if len1 >= len2 else "Model_2"
        return {
            "final_winner": winner,
            "position_bias_detected": False,
            "pass1_winner": "1" if len1 >= len2 else "2",
            "pass2_winner": "1" if len1 >= len2 else "2",
            "pass1_reasoning": "Mock evaluation based on content length heuristic.",
            "pass2_reasoning": "Mock evaluation based on content length heuristic."
        }


# =====================================================================
# Module 3: NLI Faithfulness & Hallucination Detector
# =====================================================================
class NLIHallucinationDetector:
    """Detects hallucinations using an NLI cross-encoder model."""

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base") -> None:
        self.device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
        self.active = False
        if AutoTokenizer and AutoModelForSequenceClassification:
            try:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                except Exception:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
                self.active = True
            except Exception as e:
                logger.info(f"NLI model initialization fallback to analytical mode: {e}")
                self.active = False

    def predict_entailment(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """Calculates entailment, neutral, and contradiction probabilities."""
        if not premise or not hypothesis:
            return {"contradiction": 0.0, "neutral": 1.0, "entailment": 0.0}

        if not self.active:
            # Analytical fallback based on content word overlap ratio
            stopwords = {"is", "are", "was", "were", "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or", "with", "by", "from", "inside"}
            premise_words = {w.strip(".,;:!?()\"'") for w in premise.lower().split() if w.strip(".,;:!?()\"'") not in stopwords and len(w.strip(".,;:!?()\"'")) > 1}
            hypothesis_words = {w.strip(".,;:!?()\"'") for w in hypothesis.lower().split() if w.strip(".,;:!?()\"'") not in stopwords and len(w.strip(".,;:!?()\"'")) > 1}
            
            if not hypothesis_words:
                return {"entailment": 1.0, "neutral": 0.0, "contradiction": 0.0}
                
            overlap = len(premise_words & hypothesis_words)
            ratio = float(overlap / len(hypothesis_words))
            
            entailment = round(min(1.0, ratio), 4)
            contradiction = round(max(0.0, 1.0 - ratio), 4)
            neutral = 0.0
            return {
                "contradiction": float(contradiction),
                "neutral": float(neutral),
                "entailment": float(entailment)
            }

        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

        # NLI standard logits ordering: [Contradiction, Neutral, Entailment]
        return {
            "contradiction": float(probs[0]),
            "neutral": float(probs[1]),
            "entailment": float(probs[2])
        }

    def evaluate_faithfulness(self, context: Optional[str], output_text: str) -> Dict[str, Any]:
        """Verifies sentence-level entailment against source context."""
        if not context:
            return {
                "faithfulness_score": 0.0,
                "hallucination_detected": True,
                "hallucinated_sentence_count": 1,
                "hallucinated_details": [{"sentence": output_text, "reason": "Empty source context provided."}]
            }
            
        if not output_text:
            return {
                "faithfulness_score": 1.0,
                "hallucination_detected": False,
                "hallucinated_sentence_count": 0,
                "hallucinated_details": []
            }

        sentences = [s.strip() for s in output_text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
        if not sentences:
            return {
                "faithfulness_score": 1.0,
                "hallucination_detected": False,
                "hallucinated_sentence_count": 0,
                "hallucinated_details": []
            }

        hallucinated = []
        entailment_scores = []

        for stmt in sentences:
            scores = self.predict_entailment(premise=context, hypothesis=stmt)
            entailment_scores.append(scores["entailment"])
            if scores["contradiction"] > 0.4 or scores["entailment"] < 0.3:
                hallucinated.append({"sentence": stmt, "scores": scores})

        faithfulness_score = float(np.mean(entailment_scores)) if entailment_scores else 1.0
        return {
            "faithfulness_score": round(faithfulness_score, 4),
            "hallucination_detected": len(hallucinated) > 0,
            "hallucinated_sentence_count": len(hallucinated),
            "hallucinated_details": hallucinated
        }


# =====================================================================
# Module 4: Real-Time Telemetry & Pipeline Integration Engine
# =====================================================================
@dataclass
class TelemetryRecord:
    trace_id: str
    prompt: str
    response: str
    reference: Optional[str]
    context: Optional[str]
    ttft_ms: float
    tpot_ms: float
    total_latency_ms: float
    token_count: int
    eval_metrics: Dict[str, Any]


class UnifiedEvalPipeline:
    """Integrated engine executing offline metrics, pairwise judge, hallucination detection, and OTel telemetry."""

    def __init__(self) -> None:
        self.offline_evaluator = DeterministicMetricsEvaluator()
        self.judge = PairwiseLLMJudge()
        self.hallucination_detector = NLIHallucinationDetector()

    def run_pipeline(
        self,
        prompt: str,
        response: str,
        reference: Optional[str] = None,
        context: Optional[str] = None,
        comparison_response: Optional[str] = None,
        simulated_ttft_ms: float = 45.0,
        simulated_total_latency_ms: float = 350.0
    ) -> TelemetryRecord:

        start_time = time.time()

        with tracer.start_as_current_span("LLM_Evaluation_Execution") as current_span:
            trace_id = hex(current_span.get_span_context().trace_id)
            eval_results: Dict[str, Any] = {}

            # 1. Deterministic Metrics Pass
            if reference:
                with tracer.start_as_current_span("Offline_Metrics"):
                    eval_results["deterministic"] = self.offline_evaluator.evaluate_all(response, reference)

            # 2. Faithfulness / Hallucination Detection Pass
            if context is not None:
                with tracer.start_as_current_span("Hallucination_Detection"):
                    eval_results["faithfulness"] = self.hallucination_detector.evaluate_faithfulness(context, response)

            # 3. LLM-as-a-Judge Pairwise Comparison Pass
            if comparison_response:
                with tracer.start_as_current_span("LLM_Pairwise_Judge"):
                    eval_results["pairwise_judge"] = self.judge.judge_pairwise(prompt, response, comparison_response)

            # Latency and token throughput calculations
            tokens = max(1, len(response.split()))
            tpot = (simulated_total_latency_ms - simulated_ttft_ms) / max(1, tokens - 1)

            current_span.set_attribute("llm.tokens.count", tokens)
            current_span.set_attribute("llm.latency.ttft_ms", simulated_ttft_ms)
            current_span.set_attribute("llm.latency.tpot_ms", tpot)

            record = TelemetryRecord(
                trace_id=trace_id,
                prompt=prompt,
                response=response,
                reference=reference,
                context=context,
                ttft_ms=simulated_ttft_ms,
                tpot_ms=round(tpot, 2),
                total_latency_ms=simulated_total_latency_ms,
                token_count=tokens,
                eval_metrics=eval_results
            )
            return record


if __name__ == "__main__":
    pipeline = UnifiedEvalPipeline()

    test_prompt = "What are the primary causes of global warming?"
    test_context = ("Global warming is primarily driven by human activity, notably the burning of fossil fuels "
                    "which increases heat-trapping greenhouse gas levels in Earth's atmosphere.")
    test_response = ("Global warming is caused by human activity and fossil fuel consumption. "
                     "Additionally, alien space rays heat up the ocean.")
    test_reference = "Human activities like burning fossil fuels cause global warming by increasing greenhouse gases."

    print("Running Unified LLM Evaluation Engine...")
    result = pipeline.run_pipeline(
        prompt=test_prompt,
        response=test_response,
        reference=test_reference,
        context=test_context,
        comparison_response="Global warming happens due to human greenhouse gas emissions.",
        simulated_ttft_ms=50.0,
        simulated_total_latency_ms=400.0
    )

    print("\n================ EVALUATION TELEMETRY RECORD ================")
    print(json.dumps(asdict(result), indent=2))
