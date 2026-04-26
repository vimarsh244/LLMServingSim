#!/usr/bin/env python3
"""Generate request traces from ShareGPT sessions or synthetic fixed-length traffic.

Examples:
  python dataset/sharegpt_parser.py --max-requests 750 --tokenizer-preset llama
  python dataset/sharegpt_parser.py --max-requests 1000 --tokenizer-preset phi
  python dataset/sharegpt_parser.py --max-requests 1500 --tokenizer-preset mixtral
  python dataset/sharegpt_parser.py --fixed-len --max-requests 512 --fixed-input-length 128 --fixed-output-length 512
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


TOKENIZER_PRESETS = {
    "llama": "meta-llama/Llama-3.1-8B",
    "phi": "microsoft/Phi-mini-MoE-instruct",
    "mixtral": "mistralai/Mixtral-8x7B-v0.1",
}


def _slugify(raw: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", raw.strip().lower())
    return re.sub(r"_+", "_", cleaned).strip("_")


def _rate_tag(rate: float) -> str:
    if float(rate).is_integer():
        return str(int(rate))
    return str(rate).replace(".", "p")


def _delay_tag(delay_seconds: float) -> str:
    if float(delay_seconds).is_integer():
        return str(int(delay_seconds))
    return str(delay_seconds).replace(".", "p")


def _resolve_tokenizer_name(args: argparse.Namespace) -> str:
    if args.tokenizer_name:
        return args.tokenizer_name
    return TOKENIZER_PRESETS[args.tokenizer_preset]


def _tokenizer_tag(tokenizer_name: str, tokenizer_preset: str) -> str:
    name_l = tokenizer_name.lower()
    if "llama" in name_l:
        return "llama"
    if "phi" in name_l:
        return "phi"
    if "mixtral" in name_l or "mistral" in name_l:
        return "mixtral"

    if tokenizer_preset in TOKENIZER_PRESETS:
        return tokenizer_preset

    return _slugify(Path(tokenizer_name).name)


def _default_output_name(args: argparse.Namespace, tokenizer_tag: str) -> str:
    rate = _rate_tag(args.request_per_sec)

    if args.fixed_len:
        return (
            f"fixed_in{args.fixed_input_length}_out{args.fixed_output_length}"
            f"_req{args.max_requests}_rate{rate}.jsonl"
        )

    if args.pulse:
        groups = max(1, math.ceil(args.max_requests / args.num_req_pulse))
        return (
            f"sharegpt_pulse_req{args.num_req_pulse}_n{groups}"
            f"_delay{_delay_tag(args.delay_seconds)}_{tokenizer_tag}.jsonl"
        )

    return f"sharegpt_req{args.max_requests}_rate{rate}_{tokenizer_tag}.jsonl"


def _resolve_output_path(args: argparse.Namespace, tokenizer_tag: str) -> Path:
    default_name = _default_output_name(args, tokenizer_tag)
    script_dir = Path(__file__).resolve().parent

    if args.output_path:
        out_path = Path(args.output_path)
        if not out_path.is_absolute():
            out_path = script_dir / out_path
        return out_path

    return script_dir / default_name


def _parse_sessions(raw_dataset) -> List[List[Tuple[str, str]]]:
    sessions: List[List[Tuple[str, str]]] = []

    for row in tqdm(raw_dataset, desc="Parsing ShareGPT sessions"):
        conversations = row.get("conversations", [])
        context = ""
        turns: List[Tuple[str, str]] = []

        for i in range(0, len(conversations) - 1, 2):
            if conversations[i].get("from") != "human" or conversations[i + 1].get("from") != "gpt":
                continue

            prompt = str(conversations[i].get("value", "")).strip()
            response = str(conversations[i + 1].get("value", "")).strip()
            if not prompt or not response:
                continue

            input_text = f"{context} {prompt}".strip() if context else prompt
            turns.append((input_text, response))
            context = f"{context} {prompt} {response}".strip() if context else f"{prompt} {response}".strip()

        if turns:
            sessions.append(turns)

    return sessions


def _sample_interval_ns(request_per_sec: float) -> int:
    return int(np.random.exponential(scale=1e9 / request_per_sec))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ShareGPT-based JSONL request traces")

    parser.add_argument("--dataset-name", type=str, default="shibing624/sharegpt_gpt4", help="Hugging Face dataset name")
    parser.add_argument(
        "--tokenizer-preset",
        type=str,
        choices=sorted(TOKENIZER_PRESETS.keys()),
        default="llama",
        help="Tokenizer preset used for tokenization and output naming",
    )
    parser.add_argument("--tokenizer-name", type=str, default=None, help="Override tokenizer HF model id")
    parser.add_argument("--hf-token", type=str, default=None, help="Optional Hugging Face token")

    parser.add_argument("--request-per-sec", type=float, default=10.0, help="Arrival rate in requests per second")
    parser.add_argument("--max-sessions", type=int, default=2000, help="Maximum number of ShareGPT sessions to parse")
    parser.add_argument("--max-requests", type=int, default=300, help="Maximum requests to emit")
    parser.add_argument("--first-arrival-time", type=float, default=0.0, help="First request arrival time in seconds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--room-for-decode", type=int, default=0, help="Reserved input token budget for decode")
    parser.add_argument("--max-input-length", type=int, default=None, help="Max input token length per request")
    parser.add_argument("--max-output-length", type=int, default=2048, help="Max output token length per request")
    parser.add_argument("--max-kv-length", type=int, default=2048, help="Max input+output token length")

    parser.add_argument("--fixed-len", action="store_true", help="Emit synthetic fixed-length requests")
    parser.add_argument("--fixed-input-length", type=int, default=128, help="Fixed input token count")
    parser.add_argument("--fixed-output-length", type=int, default=512, help="Fixed output token count")

    parser.add_argument("--pulse", action="store_true", help="Enable pulse arrivals")
    parser.add_argument("--num-req-pulse", type=int, default=10, help="Requests in each pulse")
    parser.add_argument("--delay-seconds", type=float, default=60.0, help="Idle delay between pulses in seconds")
    parser.add_argument("--use-poisson-in-pulse", action="store_true", help="Also sample Poisson intervals inside pulses")

    parser.add_argument("--output-path", type=str, default=None, help="Output JSONL path (default: dataset/<auto-name>)")

    args = parser.parse_args()

    if args.request_per_sec <= 0:
        raise ValueError("--request-per-sec must be > 0")
    if args.max_requests <= 0:
        raise ValueError("--max-requests must be > 0")
    if args.max_sessions <= 0:
        raise ValueError("--max-sessions must be > 0")
    if args.num_req_pulse <= 0:
        raise ValueError("--num-req-pulse must be > 0")
    if args.fixed_input_length <= 0 or args.fixed_output_length <= 0:
        raise ValueError("--fixed-input-length and --fixed-output-length must be > 0")
    if args.max_output_length <= 0 or args.max_kv_length <= 0:
        raise ValueError("--max-output-length and --max-kv-length must be > 0")
    if args.room_for_decode < 0:
        raise ValueError("--room-for-decode must be >= 0")

    return args


def main() -> None:
    args = _parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    random.seed(args.seed)
    np.random.seed(args.seed)

    tokenizer_name = _resolve_tokenizer_name(args)
    tokenizer_tag = _tokenizer_tag(tokenizer_name, args.tokenizer_preset)
    output_path = _resolve_output_path(args, tokenizer_tag)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    max_input_length = args.max_input_length
    if max_input_length is None:
        max_input_length = max(1, args.max_kv_length - args.room_for_decode)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    sessions: List[List[Tuple[str, str]]] = []
    session_indices: List[int] = []
    if not args.fixed_len:
        raw_dataset = load_dataset(args.dataset_name, split="train")
        keep = min(args.max_sessions, len(raw_dataset))
        raw_dataset = raw_dataset.select(range(keep))
        sessions = _parse_sessions(raw_dataset)
        session_indices = [0] * len(sessions)

        if not sessions:
            raise RuntimeError("No valid conversation turns found in dataset.")

    request_count = 0
    skipped_for_length = 0
    time_offset_ns = int(args.first_arrival_time * 1_000_000_000)

    with open(output_path, "w", encoding="utf-8") as fout:
        while request_count < args.max_requests:
            if args.fixed_len:
                vocab_size = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") and tokenizer.vocab_size else 32000
                input_tokens = [random.randint(0, vocab_size - 1) for _ in range(args.fixed_input_length)]
                output_tokens = [random.randint(0, vocab_size - 1) for _ in range(args.fixed_output_length)]
            else:
                available_sessions = [
                    i for i, idx in enumerate(session_indices) if idx < len(sessions[i])
                ]
                if not available_sessions:
                    break

                sid = random.choice(available_sessions)
                input_text, output_text = sessions[sid][session_indices[sid]]
                session_indices[sid] += 1

                input_tokens = tokenizer(input_text, add_special_tokens=False)["input_ids"]
                output_tokens = tokenizer(output_text, add_special_tokens=False)["input_ids"]

                total_tokens = len(input_tokens) + len(output_tokens)
                too_long = (
                    len(input_tokens) > max_input_length
                    or len(output_tokens) > args.max_output_length
                    or total_tokens > args.max_kv_length
                )
                if too_long:
                    skipped_for_length += 1
                    continue

            if args.pulse and request_count % args.num_req_pulse == 0 and request_count > 0:
                time_offset_ns += int(args.delay_seconds * 1_000_000_000)
            elif not args.pulse or args.use_poisson_in_pulse:
                time_offset_ns += _sample_interval_ns(args.request_per_sec)

            record = {
                "input_toks": len(input_tokens),
                "output_toks": len(output_tokens),
                "arrival_time_ns": time_offset_ns,
                "input_tok_ids": input_tokens,
                "output_tok_ids": output_tokens,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            request_count += 1

    print("=" * 88)
    print("ShareGPT Dataset Generator")
    print(f"Tokenizer:         {tokenizer_name}")
    print(f"Tokenizer tag:     {tokenizer_tag}")
    print(f"Output file:       {output_path}")
    print(f"Requested entries: {args.max_requests}")
    print(f"Written entries:   {request_count}")
    if skipped_for_length > 0:
        print(f"Skipped (length):  {skipped_for_length}")
    if request_count < args.max_requests and not args.fixed_len:
        print("Warning: dataset exhausted before reaching requested size.")
        print("Increase --max-sessions or loosen max-length constraints.")
    print("=" * 88)


if __name__ == "__main__":
    main()
