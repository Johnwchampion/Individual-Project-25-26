

import json
import os
from glob import glob
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ChatRecord:
	id: str
	condition: str
	messages: List[Dict]


@dataclass(frozen=True)
class ChatPair:
	base_id: str
	with_context: ChatRecord
	no_context: ChatRecord


# Utility functions for loading and processing the dataset

def load_jsonl(path: str) -> List[ChatRecord]:
	if os.path.isdir(path):
		candidates = sorted(glob(os.path.join(path, "*.jsonl")))
		if len(candidates) != 1:
			raise ValueError(
				f"Expected exactly one .jsonl in directory {path}, found {len(candidates)}."
			)
		path = candidates[0]
	records: List[ChatRecord] = []
	with open(path, "r", encoding="utf-8") as handle:
		for line in handle:
			line = line.strip()
			if not line:
				continue
			obj = json.loads(line)
			records.append(
				ChatRecord(
					id=obj["id"],
					condition=obj["condition"],
					messages=obj["messages"],
				)
			)
	return records


def validate_alternating_structure(records: List[ChatRecord]) -> None:
	if len(records) % 2 != 0:
		raise ValueError("Record count must be even for strict alternation.")
	for i in range(0, len(records), 2):
		first = records[i]
		second = records[i + 1]
		if first.condition not in {"with_context", "no_context"}:
			raise ValueError(f"Invalid condition at index {i}: {first.condition}")
		if second.condition not in {"with_context", "no_context"}:
			raise ValueError(f"Invalid condition at index {i + 1}: {second.condition}")
		if first.condition != "with_context":
			raise ValueError(
				f"Alternation break at index {i}: expected with_context, got {first.condition}"
			)
		if second.condition != "no_context":
			raise ValueError(
				f"Alternation break at index {i + 1}: expected no_context, got {second.condition}"
			)


def group_into_pairs(records: List[ChatRecord]) -> List[ChatPair]:
	validate_alternating_structure(records)
	pairs: List[ChatPair] = []
	for i in range(0, len(records), 2):
		with_context = records[i]
		no_context = records[i + 1]
		pairs.append(
			ChatPair(
				base_id=with_context.id.rsplit("_", 1)[0],
				with_context=with_context,
				no_context=no_context,
			)
		)
	return pairs


def sample_pairs(pairs: List[ChatPair], n: int) -> List[ChatPair]:
	if n < 0:
		raise ValueError("Sample size must be non-negative.")
	if n > len(pairs):
		raise ValueError("Sample size cannot exceed number of pairs.")
	if n == len(pairs):
		return list(pairs)
	return pairs[:n]


# ---------------------------------------------------------------------------
# Safety pipeline types and loaders
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SafetyPair:
	base_id: str
	unsafe: ChatRecord
	safe: ChatRecord


def group_into_safety_pairs(records: List[ChatRecord]) -> List[SafetyPair]:
	"""
	Group alternating unsafe/safe ChatRecords into SafetyPair objects.

	Expects records in strict interleaved order: unsafe, safe, unsafe, safe, ...
	This matches the output format of prepare_safedata.py, which writes an
	unsafe record immediately followed by its paired safe record for the same prompt.
	"""
	if len(records) % 2 != 0:
		raise ValueError("Record count must be even for safety pair grouping.")

	pairs: List[SafetyPair] = []
	for i in range(0, len(records), 2):
		unsafe_rec = records[i]
		safe_rec = records[i + 1]

		if unsafe_rec.condition != "unsafe":
			raise ValueError(
				f"Expected 'unsafe' at index {i}, got '{unsafe_rec.condition}'"
			)
		if safe_rec.condition != "safe":
			raise ValueError(
				f"Expected 'safe' at index {i + 1}, got '{safe_rec.condition}'"
			)

		pairs.append(
			SafetyPair(
				base_id=unsafe_rec.id.rsplit("_", 1)[0],
				unsafe=unsafe_rec,
				safe=safe_rec,
			)
		)

	return pairs
