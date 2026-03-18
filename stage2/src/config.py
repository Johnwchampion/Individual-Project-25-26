import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../stage1/src"))

from config import (
    SAFETY_MODEL_NAME, CACHE_DIR,
    DATA_DIR, SAFETY_DATA_DIR,
    RD_FAITH_PATH, RD_FAITH_LOGITS_PATH,
    RD_SAFETY_PATH, RD_SAFETY_LOGITS_PATH,
)

# Candidate selection
CANDIDATE_N = 10           # top-N per metric before intersection; varied in experiments

# Intervention
SOFT_STRENGTH = 30.0       # logit penalty for soft suppression; varied in experiments

# Generation
MAX_NEW_TOKENS = 150

# Results output
RESULTS_DIR = "/scratch/sc23jc3/stage2_results"

__all__ = [
    "SAFETY_MODEL_NAME", "CACHE_DIR",
    "DATA_DIR", "SAFETY_DATA_DIR",
    "RD_FAITH_PATH", "RD_FAITH_LOGITS_PATH",
    "RD_SAFETY_PATH", "RD_SAFETY_LOGITS_PATH",
    "CANDIDATE_N", "SOFT_STRENGTH",
    "MAX_NEW_TOKENS", "RESULTS_DIR",
]
