# --- Faithfulness pipeline (SQuAD) ---
MODEL_NAME = "deepseek-ai/DeepSeek-V2-Lite"
CACHE_DIR = "/scratch/sc23jc3/cache"
DATA_DIR = "/scratch/sc23jc3/squad_prepared"
OUTPUT_DIR = "/scratch/sc23jc3/squad_results"

# --- Safety pipeline (BeaverTails) ---
# Uses the Chat model so routing is measured over safety-trained weights.
SAFETY_MODEL_NAME = "deepseek-ai/DeepSeek-V2-Lite-Chat"
SAFETY_DATA_DIR = "/scratch/sc23jc3/beavertails_prepared"
SAFETY_OUTPUT_DIR = "/scratch/sc23jc3/beavertails_results"

# Minimum number of length-matched response tokens required to include a pair.
# Pairs shorter than this are skipped to avoid noisy single-token routing counts.
MIN_RESPONSE_TOKENS = 5

__all__ = [
    "MODEL_NAME", "CACHE_DIR", "DATA_DIR", "OUTPUT_DIR",
    "SAFETY_MODEL_NAME", "SAFETY_DATA_DIR", "SAFETY_OUTPUT_DIR",
    "MIN_RESPONSE_TOKENS",
]
