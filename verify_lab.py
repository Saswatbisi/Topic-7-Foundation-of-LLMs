"""Automated Verification Suite for LLM Evaluation Engine."""

import unittest
import json
from eval_engine import UnifiedEvalPipeline, DeterministicMetricsEvaluator, PairwiseLLMJudge, NLIHallucinationDetector


SYNTHETIC_BENCHMARK_DATASET = [
    {
        "id": "TC-PASS-01",
        "prompt": "What is the capital of France?",
        "context": "Paris is the capital city and most populous municipality of France.",
        "response": "Paris is the capital of France.",
        "reference": "The capital of France is Paris.",
        "expected_faithfulness": True,
        "min_rouge_l": 0.50
    },
    {
        "id": "TC-FAIL-02",
        "prompt": "Where is the headquarter of NASA located?",
        "context": "NASA headquarters is located in Washington, D.C.",
        "response": "NASA headquarters is located on the Surface of Mars inside Crater Alpha.",
        "reference": "Washington, D.C. houses NASA headquarters.",
        "expected_faithfulness": False,
        "max_rouge_l": 0.50
    },
    {
        "id": "TC-PASS-03",
        "prompt": "What is photosynthesis?",
        "context": "Photosynthesis converts light energy into chemical energy, producing oxygen and carbohydrates like glucose.",
        "response": "Photosynthesis produces oxygen and carbohydrates such as glucose from light energy.",
        "reference": "The outputs of photosynthesis are oxygen and glucose.",
        "expected_faithfulness": True,
        "min_rouge_l": 0.50
    },
    {
        "id": "TC-FAIL-04",
        "prompt": "What company built the Saturn V rocket?",
        "context": "The Saturn V rocket was manufactured by Boeing, North American Aviation, and Douglas Aircraft Company for NASA.",
        "response": "The Saturn V rocket was constructed entirely by Tesla in Gigafactory Texas.",
        "reference": "Boeing and North American Aviation manufactured the Saturn V rocket.",
        "expected_faithfulness": False,
        "max_rouge_l": 0.40
    },
    {
        "id": "TC-PASS-05",
        "prompt": "What is the speed of light in vacuum?",
        "context": "The speed of light in vacuum is exactly 299,792,458 meters per second.",
        "response": "Light travels at 299,792,458 meters per second in a vacuum.",
        "reference": "Light speed in vacuum is approximately 299,792,458 m/s.",
        "expected_faithfulness": True,
        "min_rouge_l": 0.30
    },
    {
        "id": "TC-FAIL-06",
        "prompt": "Who discovered Penicillin?",
        "context": "Alexander Fleming discovered penicillin in 1928 at St. Mary's Hospital, London.",
        "response": "Penicillin was discovered by Albert Einstein during his research on quantum physics in Berlin.",
        "reference": "Alexander Fleming discovered penicillin.",
        "expected_faithfulness": False,
        "max_rouge_l": 0.50
    },
    {
        "id": "TC-PASS-07",
        "prompt": "What chemical formula represents water?",
        "context": "Water is a chemical compound consisting of two hydrogen atoms bonded to one oxygen atom, forming H2O.",
        "response": "The chemical formula of water is H2O.",
        "reference": "H2O is the chemical symbol for water.",
        "expected_faithfulness": True,
        "min_rouge_l": 0.40
    },
    {
        "id": "TC-FAIL-08",
        "prompt": "What is the boiling point of water at sea level?",
        "context": "At standard atmospheric pressure, water boils at 100 degrees Celsius or 212 degrees Fahrenheit.",
        "response": "Water boils at 500 degrees Celsius under standard atmospheric sea level conditions.",
        "reference": "Water boils at 100 C at sea level.",
        "expected_faithfulness": False,
        "max_rouge_l": 0.60
    },
    {
        "id": "TC-PASS-09",
        "prompt": "What is the largest planet in our solar system?",
        "context": "Jupiter is the fifth planet from the Sun and the largest in the Solar System.",
        "response": "Jupiter is the largest planet in the Solar System.",
        "reference": "The largest planet is Jupiter.",
        "expected_faithfulness": True,
        "min_rouge_l": 0.50
    },
    {
        "id": "TC-FAIL-10",
        "prompt": "Who painted the Mona Lisa?",
        "context": "The Mona Lisa is a half-length portrait painting by Italian artist Leonardo da Vinci.",
        "response": "The Mona Lisa was painted by Vincent van Gogh in Arles.",
        "reference": "Leonardo da Vinci painted the Mona Lisa.",
        "expected_faithfulness": False,
        "max_rouge_l": 0.50
    },
    {
        "id": "TC-PASS-11",
        "prompt": "What is the primary function of hemoglobin?",
        "context": "Hemoglobin is an iron-containing protein in red blood cells that transports oxygen throughout the body.",
        "response": "Hemoglobin transports oxygen in red blood cells.",
        "reference": "Hemoglobin carries oxygen in the blood.",
        "expected_faithfulness": True,
        "min_rouge_l": 0.40
    },
    {
        "id": "TC-FAIL-12",
        "prompt": "Which element has atomic number 1?",
        "context": "Hydrogen is the chemical element with the symbol H and atomic number 1.",
        "response": "Uranium is the element with atomic number 1 in the periodic table.",
        "reference": "Hydrogen has atomic number 1.",
        "expected_faithfulness": False,
        "max_rouge_l": 0.60
    },
    {
        "id": "TC-PASS-13",
        "prompt": "What year did Apollo 11 land on the Moon?",
        "context": "Apollo 11 landed on the Moon on July 20, 1969.",
        "response": "Apollo 11 landed on the Moon in July 1969.",
        "reference": "The Moon landing occurred in 1969.",
        "expected_faithfulness": True,
        "min_rouge_l": 0.40
    },
    {
        "id": "TC-FAIL-14",
        "prompt": "What is DNA's structure?",
        "context": "DNA is composed of two polynucleotide chains that coil around each other to form a double helix.",
        "response": "DNA is structured as a triple helix composed of metallic carbohydrates.",
        "reference": "DNA is a double helix structure.",
        "expected_faithfulness": False,
        "max_rouge_l": 0.50
    },
    {
        "id": "TC-PASS-15",
        "prompt": "What causes oceanic tides?",
        "context": "Oceanic tides are caused by gravitational pull exerted by the Moon and Sun on Earth's oceans.",
        "response": "Tides are driven by the gravitational pull of the Moon and Sun.",
        "reference": "The Moon's gravity causes ocean tides.",
        "expected_faithfulness": True,
        "min_rouge_l": 0.40
    }
]


class TestLLMEvaluationPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pipeline = UnifiedEvalPipeline()
        cls.evaluator = DeterministicMetricsEvaluator()
        cls.judge = PairwiseLLMJudge()
        cls.detector = NLIHallucinationDetector()

    def test_01_faithful_payload_passing(self):
        """Validates that accurate context-backed outputs pass hallucination gates."""
        payload = SYNTHETIC_BENCHMARK_DATASET[0]
        record = self.pipeline.run_pipeline(
            prompt=payload["prompt"],
            response=payload["response"],
            reference=payload["reference"],
            context=payload["context"]
        )
        metrics = record.eval_metrics
        self.assertGreater(metrics["deterministic"]["rougeL_f1"], payload["min_rouge_l"])
        self.assertFalse(metrics["faithfulness"]["hallucination_detected"])

    def test_02_hallucinated_payload_flagging(self):
        """Validates that conflicting, hallucinated context outputs are flagged."""
        payload = SYNTHETIC_BENCHMARK_DATASET[1]
        record = self.pipeline.run_pipeline(
            prompt=payload["prompt"],
            response=payload["response"],
            reference=payload["reference"],
            context=payload["context"]
        )
        metrics = record.eval_metrics
        self.assertTrue(metrics["faithfulness"]["hallucination_detected"])
        self.assertGreater(metrics["faithfulness"]["hallucinated_sentence_count"], 0)

    def test_03_pairwise_judge_bias_mitigation(self):
        """Validates that pairwise evaluation outputs structure correctly."""
        prompt = "Explain quantum computing briefly."
        resp1 = "Quantum computing uses qubits utilizing superposition and entanglement principles."
        resp2 = "Quantum computing is just faster classical computers using standard transistor logic."

        record = self.pipeline.run_pipeline(
            prompt=prompt,
            response=resp1,
            comparison_response=resp2
        )
        judge_results = record.eval_metrics["pairwise_judge"]
        self.assertIn("final_winner", judge_results)
        self.assertIn("position_bias_detected", judge_results)
        self.assertIsInstance(judge_results["position_bias_detected"], bool)

    def test_04_synthetic_15_sample_benchmark(self):
        """Validates the full 15-sample synthetic benchmark dataset."""
        passed_count = 0
        for tc in SYNTHETIC_BENCHMARK_DATASET:
            rec = self.pipeline.run_pipeline(
                prompt=tc["prompt"],
                response=tc["response"],
                reference=tc["reference"],
                context=tc["context"]
            )
            detected = rec.eval_metrics["faithfulness"]["hallucination_detected"]
            expected_flag = not tc["expected_faithfulness"]
            if detected == expected_flag:
                passed_count += 1
                
        accuracy = passed_count / len(SYNTHETIC_BENCHMARK_DATASET)
        print(f"\nSynthetic Benchmark Accuracy: {passed_count}/{len(SYNTHETIC_BENCHMARK_DATASET)} ({accuracy*100:.1f}%)")
        self.assertGreaterEqual(accuracy, 0.70)

    def test_05_edge_cases_handling(self):
        """Tests handling of empty context, null values, and empty strings."""
        # Empty context
        rec1 = self.pipeline.run_pipeline(
            prompt="Test",
            response="Test response",
            context=None
        )
        self.assertNotIn("faithfulness", rec1.eval_metrics)

        # Empty response string
        rec2 = self.pipeline.run_pipeline(
            prompt="Test",
            response="",
            reference="Test ref",
            context="Test context"
        )
        self.assertEqual(rec2.eval_metrics["deterministic"]["exact_match"], 0.0)
        self.assertFalse(rec2.eval_metrics["faithfulness"]["hallucination_detected"])


if __name__ == "__main__":
    print("Executing Pipeline Assertions & Validation Suite...\n")
    unittest.main(verbosity=2)
