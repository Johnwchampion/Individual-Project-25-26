# Faithfulness pipeline (SQuAD)
MODEL_NAME = "deepseek-ai/DeepSeek-V2-Lite"
CACHE_DIR = "/scratch/sc23jc3/cache"
DATA_DIR = "/scratch/sc23jc3/squad_prepared"
OUTPUT_DIR = "/scratch/sc23jc3/squad_results"

#
# Uses the Chat model so routing is measured over safety-trained weights.
SAFETY_MODEL_NAME = "deepseek-ai/DeepSeek-V2-Lite-Chat"
SAFETY_DATA_DIR = "/scratch/sc23jc3/beavertails_prepared"
SAFETY_OUTPUT_DIR = "/scratch/sc23jc3/beavertails_results"

# Minimum number of length-matched response tokens required to include a pair.
# Pairs shorter than this are skipped to avoid noisy single-token routing counts.
MIN_RESPONSE_TOKENS = 5

# RD output paths (loaded by Stage 2)
RD_FAITH_PATH         = "/scratch/sc23jc3/results/rd_faithfulness.json"
RD_FAITH_LOGITS_PATH  = "/scratch/sc23jc3/results/rd_faithfulness_logits.json"
RD_SAFETY_PATH        = "/scratch/sc23jc3/results/rd_safety.json"
RD_SAFETY_LOGITS_PATH = "/scratch/sc23jc3/results/rd_safety_logits.json"

__all__ = [
    "MODEL_NAME", "CACHE_DIR", "DATA_DIR", "OUTPUT_DIR",
    "SAFETY_MODEL_NAME", "SAFETY_DATA_DIR", "SAFETY_OUTPUT_DIR",
    "MIN_RESPONSE_TOKENS",
    "RD_FAITH_PATH", "RD_FAITH_LOGITS_PATH",
    "RD_SAFETY_PATH", "RD_SAFETY_LOGITS_PATH",
]
