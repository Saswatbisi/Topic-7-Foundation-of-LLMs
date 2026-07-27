"""Pre-flight sanity check diagnostic for Topic 7 dependencies."""
import sys

def run_diagnostics():
    print("Running system dependency checks for Topic 7...")
    modules = [
        ("torch", "PyTorch"),
        ("transformers", "HuggingFace Transformers"),
        ("nltk", "NLTK"),
        ("rouge_score", "ROUGE Score"),
        ("bert_score", "BERTScore"),
        ("openai", "OpenAI Client"),
        ("opentelemetry", "OpenTelemetry API"),
    ]
    
    passed = True
    for mod, name in modules:
        try:
            __import__(mod)
            print(f"  [OK] {name} ({mod}) loaded successfully.")
        except ImportError as e:
            print(f"  [FAIL] {name} ({mod}) failed to load: {e}")
            passed = False
            
    if passed:
        print("\nENV SETUP COMPLETE: All modules imported successfully.")
    else:
        print("\nENV SETUP WARNING: Some modules are missing. Fallback logic will be utilized where applicable.")

if __name__ == "__main__":
    run_diagnostics()
