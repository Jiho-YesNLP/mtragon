
# """
# Question Rewriter
# ------------------

# Reads questions from a JSONL file and uses the OpenAI API to produce
# multiple rewritten queries optimized for document retrieval, including:
# - spell correction
# - query expansion (synonyms, related terms, common variants)
# - acronym expansion
# - light coreference resolution (replace pronouns with explicit entities)

# Input (JSONL):
# - Flexible schema. The script will attempt to extract the question text from one of:
# 	- item["question"]
# 	- item["query"]
# 	- item["text"] (if conversational, the last line that starts with "user:" is used)

# Output (JSONL):
# - One line per input item with fields:
# 	{
# 		"task_id": str,
# 		"original": str,
# 		"rewrites": List[str]
# 	}

# Usage example:
# 	python scripts/question_rewriter.py \
# 		--input data/raw/human/questions.jsonl \
# 		--output outputs/rewritten_questions.jsonl \
# 		--model gpt-4o-mini --num-rewrites 3

# Requires environment variable OPENAI_API_KEY to be set.
# """

# from __future__ import annotations

# import argparse
# import json
# import os
# import re
# import sys
# import time
# from typing import Any, Dict, Iterable, List, Optional


# def _extract_last_user_utterance(text: str) -> str:
# 	"""Extract the last user utterance from a conversation-like text.

# 	Assumes lines like:
# 	  user: ...
# 	  agent: ...
# 	Returns the text after the final "user:" prefix if found; otherwise returns the full text.
# 	"""
# 	if not isinstance(text, str):
# 		return ""
# 	lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
# 	# Find the last line that starts with 'user:' (case-insensitive)
# 	for ln in reversed(lines):
# 		if ln.lower().startswith("user:"):
# 			# common formats: "user: ..." or "user - ..."; strip the prefix and surrounding whitespace
# 			return re.sub(r"^user\s*[:\-]\s*", "", ln, flags=re.IGNORECASE).strip()
# 	return text.strip()


# def _extract_question(item: Dict[str, Any]) -> str:
# 	"""Robust extraction of a question string from a JSON item."""
# 	if not isinstance(item, dict):
# 		return ""
# 	# Prefer explicit fields if present
# 	for key in ("question", "query"):
# 		if key in item and isinstance(item[key], str) and item[key].strip():
# 			return item[key].strip()
# 	# Fallback to "text" which may contain a conversation
# 	if "text" in item and isinstance(item["text"], str) and item["text"].strip():
# 		return _extract_last_user_utterance(item["text"]) or item["text"].strip()
# 	# Ultimate fallback: try a generic key
# 	for key, val in item.items():
# 		if isinstance(val, str) and val.strip() and any(k in key.lower() for k in ["q", "question", "query", "utterance", "prompt"]):
# 			return val.strip()
# 	return ""


# def _extract_task_id(item: Dict[str, Any], idx: int) -> str:
# 	"""Resolve a stable id from the item or fall back to a sequential index."""
# 	for key in ("task_id", "_id", "id"):
# 		if key in item and isinstance(item[key], str) and item[key].strip():
# 			return item[key].strip()
# 	for key in ("task_id", "_id", "id"):
# 		if key in item and isinstance(item[key], (int, float)):
# 			return str(item[key])
# 	return f"item-{idx}"


# def build_system_prompt() -> str:
# 	return (
# 		"You are an expert query engineer helping a retrieval system. "
# 		"Rewrite the user's question into multiple clear, retrieval-optimized queries. "
# 		"Do the following for each rewrite: "
# 		"1) Fix spelling and grammar while preserving intent. "
# 		"2) Expand acronyms to their most common meanings if appropriate (keep original acronym too when helpful). "
# 		"3) Expand with synonyms and common variants for key terms. "
# 		"4) Resolve coreferences: replace pronouns or vague mentions with explicit entities from the question or brief context. "
# 		"5) Keep queries concise and keyword-rich (no extra narration). "
# 		"6) Avoid hallucinating facts that contradict the user's text. "
# 		"Return a compact JSON object with an array 'rewrites' only."
# 	)


# def build_user_prompt(question: str, num_rewrites: int) -> str:
# 	# Provide a small schema reminder to steer structured output
# 	schema_hint = {
# 		"rewrites": ["string", "..."],
# 	}
# 	return (
# 		"Question:\n" + question.strip() + "\n\n"
# 		f"Please produce {num_rewrites} diverse rewrites optimized for document retrieval. "
# 		"Output strictly as minified JSON matching this schema: "
# 		+ json.dumps(schema_hint, separators=(",", ":"))
# 	)


# def _ensure_openai_client():
# 	"""Import and initialize the OpenAI client, ensuring API key is set."""
# 	api_key = os.getenv("OPENAI_API_KEY")
# 	if not api_key:
# 		raise EnvironmentError(
# 			"OPENAI_API_KEY is not set. Please export your API key before running."
# 		)
# 	try:
# 		from openai import OpenAI  # SDK v1+
# 	except Exception as e:  # pragma: no cover
# 		raise ImportError(
# 			"Failed to import the OpenAI SDK. Install it with 'pip install openai'."
# 		) from e
# 	return OpenAI(api_key=api_key)


# def call_openai_rewriter(
# 	client,
# 	model: str,
# 	question: str,
# 	num_rewrites: int,
# 	max_retries: int = 5,
# 	timeout_s: float = 30.0,
# ) -> List[str]:
# 	"""Call the OpenAI model to generate rewrites for a single question.

# 	Retries with exponential backoff on transient failures.
# 	"""
# 	system_prompt = build_system_prompt()
# 	user_prompt = build_user_prompt(question, num_rewrites)

# 	backoff = 1.0
# 	last_err: Optional[BaseException] = None
# 	for _ in range(max_retries):
# 		try:
# 			resp = client.chat.completions.create(
# 				model=model,
# 				temperature=0.3,
# 				messages=[
# 					{"role": "system", "content": system_prompt},
# 					{"role": "user", "content": user_prompt},
# 				],
# 				timeout=timeout_s,
# 			)
# 			content = resp.choices[0].message.content if resp.choices else ""
# 			rewrites = parse_rewrites_from_content(content)
# 			if not rewrites:
# 				return [question.strip()]
# 			# Limit to requested count and deduplicate while preserving order
# 			uniq: List[str] = []
# 			seen = set()
# 			for r in rewrites:
# 				r_ = r.strip()
# 				if r_ and r_ not in seen:
# 					uniq.append(r_)
# 					seen.add(r_)
# 				if len(uniq) >= num_rewrites:
# 					break
# 			return uniq or [question.strip()]
# 		except Exception as e:
# 			last_err = e
# 			time.sleep(backoff)
# 			backoff = min(backoff * 2, 30)

# 	if last_err:
# 		sys.stderr.write(f"OpenAI error after {max_retries} attempts: {last_err}\n")
# 	return [question.strip()]


# def parse_rewrites_from_content(content: str) -> List[str]:
# 	"""Parse a list of rewrites from model content.

# 	Primary: parse strict JSON object {"rewrites": [...]}.
# 	Fallbacks: extract JSON code block, or numbered/bulleted lines.
# 	"""
# 	if not content:
# 		return []
# 	text = content.strip()

# 	# Try direct JSON first
# 	try:
# 		obj = json.loads(text)
# 		if isinstance(obj, dict) and isinstance(obj.get("rewrites"), list):
# 			return [str(x) for x in obj["rewrites"]]
# 	except Exception:
# 		pass

# 	# Try to find a JSON code block
# 	m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
# 	if m:
# 		try:
# 			obj = json.loads(m.group(1))
# 			if isinstance(obj, dict) and isinstance(obj.get("rewrites"), list):
# 				return [str(x) for x in obj["rewrites"]]
# 		except Exception:
# 			pass

# 	# Fallback: parse bullet/numbered lists
# 	lines = [ln.strip("- •*\t ") for ln in text.splitlines() if ln.strip()]
# 	cand: List[str] = []
# 	for ln in lines:
# 		ln = re.sub(r"^\(?\d+\)?[\.:]\s*", "", ln)
# 		if 1 <= len(ln) <= 256:
# 			cand.append(ln)
# 	cand = [re.sub(r"\s+", " ", c).strip() for c in cand]
# 	out: List[str] = []
# 	seen = set()
# 	for c in cand:
# 		if c and c not in seen:
# 			out.append(c)
# 			seen.add(c)
# 	return out[:10]


# def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
# 	with open(path, "r", encoding="utf-8") as f:
# 		for i, line in enumerate(f):
# 			line = line.strip()
# 			if not line:
# 				continue
# 			try:
# 				yield json.loads(line)
# 			except json.JSONDecodeError as e:
# 				sys.stderr.write(f"Skipping malformed JSON at line {i+1}: {e}\n")


# def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
# 	os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
# 	with open(path, "w", encoding="utf-8") as f:
# 		for row in rows:
# 			f.write(json.dumps(row, ensure_ascii=False) + "\n")


# def main():
# 	parser = argparse.ArgumentParser(description="Rewrite questions for retrieval using OpenAI.")
# 	parser.add_argument("--input", type=str, default="data/raw/human/questions.jsonl", help="Path to input JSONL of questions")
# 	parser.add_argument("--output", type=str, default="outputs/rewritten_questions.jsonl", help="Where to write JSONL output")
# 	parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model name")
# 	parser.add_argument("--num-rewrites", type=int, default=3, help="Number of rewrites to generate per question")
# 	parser.add_argument("--max", dest="max_items", type=int, default=0, help="Process at most this many items (0 = all)")
# 	parser.add_argument("--dry-run", action="store_true", help="Do not call API; emit simple normalized versions for testing")
# 	args = parser.parse_args()

# 	input_path = args.input
# 	output_path = args.output
# 	model = args.model
# 	n_rewrites = max(1, int(args.num_rewrites))
# 	max_items = int(args.max_items or 0)

# 	items = list(read_jsonl(input_path))
# 	if max_items > 0:
# 		items = items[:max_items]

# 	if args.dry_run:
# 		def fake_rewrites(q: str) -> List[str]:
# 			base = re.sub(r"\s+", " ", q).strip()
# 			return [base] + [f"{base} ({i})" for i in range(1, n_rewrites)]

# 		rows = []
# 		for idx, item in enumerate(items):
# 			q = _extract_question(item)
# 			tid = _extract_task_id(item, idx)
# 			if not q:
# 				continue
# 			rows.append({
# 				"task_id": tid,
# 				"original": q,
# 				"rewrites": fake_rewrites(q)[:n_rewrites],
# 			})
# 		write_jsonl(output_path, rows)
# 		print(f"[dry-run] Wrote {len(rows)} items to {output_path}")
# 		return

# 	client = _ensure_openai_client()

# 	rows = []
# 	for idx, item in enumerate(items):
# 		q = _extract_question(item)
# 		tid = _extract_task_id(item, idx)
# 		if not q:
# 			sys.stderr.write(f"Skipping item without recognizable question at index {idx}\n")
# 			continue
# 		rewrites = call_openai_rewriter(client, model=model, question=q, num_rewrites=n_rewrites)
# 		rows.append({
# 			"task_id": tid,
# 			"original": q,
# 			"rewrites": rewrites,
# 		})
# 		# Flush incrementally to avoid losing progress on long runs
# 		if idx % 20 == 0 and idx > 0:
# 			write_jsonl(output_path, rows)

# 	write_jsonl(output_path, rows)
# 	print(f"Wrote {len(rows)} items to {output_path}")

import os
import openai
from dotenv import load_dotenv
load_dotenv(override=True)


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key,
                           base_url="https://api.theparley.org")
    try:
        client.models.list()
    except openai.AuthenticationError:
        print("API key is invalid.")
    else:
        print("API key is valid.")


if __name__ == "__main__":
    main()
