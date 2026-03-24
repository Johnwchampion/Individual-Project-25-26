import importlib.util
import os

_s1_path = os.path.join(os.path.dirname(__file__), "../../stage1/src/config.py")
_spec = importlib.util.spec_from_file_location("_stage1_config", _s1_path)
_s1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_s1)

SAFETY_MODEL_NAME    = _s1.SAFETY_MODEL_NAME
CACHE_DIR            = _s1.CACHE_DIR
DATA_DIR             = _s1.DATA_DIR
SAFETY_DATA_DIR      = _s1.SAFETY_DATA_DIR
RD_FAITH_PATH        = _s1.RD_FAITH_PATH
RD_FAITH_LOGITS_PATH = _s1.RD_FAITH_LOGITS_PATH
RD_SAFETY_PATH       = _s1.RD_SAFETY_PATH
RD_SAFETY_LOGITS_PATH = _s1.RD_SAFETY_LOGITS_PATH

# Candidate selection
CANDIDATE_N = 3            # top-N per metric before intersection; varied in experiments

# Intervention
SOFT_STRENGTH = 0.5        # logit shift scale for soft pre-selection; empirically validated

# Generation
MAX_NEW_TOKENS = 150

# Results output
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

__all__ = [
    "SAFETY_MODEL_NAME", "CACHE_DIR",
    "DATA_DIR", "SAFETY_DATA_DIR",
    "RD_FAITH_PATH", "RD_FAITH_LOGITS_PATH",
    "RD_SAFETY_PATH", "RD_SAFETY_LOGITS_PATH",
    "CANDIDATE_N", "SOFT_STRENGTH",
    "MAX_NEW_TOKENS", "RESULTS_DIR",
]
