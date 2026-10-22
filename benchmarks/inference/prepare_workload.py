"""Prepare fixed token workloads for correctness checks and inference benchmarks."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=project / "artifacts/models/qwen2.5-0.5b"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=project / "artifacts/inference/inputs"
    )
    args = parser.parse_args()
    model = args.model if args.model.is_absolute() else project / args.model
    root = (
        args.output_dir if args.output_dir.is_absolute() else project / args.output_dir
    )
    for name in ("prompts.jsonl", "bench-input.jsonl", "tokenizer-info.json"):
        if (root / name).exists():
            raise FileExistsError(
                f"Reuse existing inputs or choose a new --output-dir: {root / name}"
            )
    root.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    texts = [
        "Explain KV cache in one sentence.",
        "What is a matrix multiplication?",
        "Name two differences between prefill and decode.",
        "用一句话解释显存带宽。",
        "为什么 GQA 可以减少 KV Cache？",
        "Write a Python function add(a, b).",
        "Complete this Python code: def square(x):\n    return",
        "List three Linux commands for inspecting processes.",
        "What is 17 + 25? Answer briefly.",
        "Sort these integers: 8, 3, 5, 1.",
        "Summarize: A cache stores reusable results to avoid repeated work.",
        "Translate into English: 连续批处理提高了 GPU 的利用效率。",
        "Return JSON with a single key named status and value ok.",
        "Explain the word throughput in Chinese and English.",
        "What are the next two numbers: 2, 4, 6, 8?",
        "",
        "Explain attention.   \n",
        "Remember this data: " + "alpha beta gamma delta. " * 64 + "What is repeated?",
        "Copy exactly: CACHE_BOUNDARY_123",
        "Reply only with OK.",
    ]
    assert len(texts) == 20
    rows = []
    for index, text in enumerate(texts, 1):
        messages = [{"role": "user", "content": text}]
        if index == 15:
            messages = [
                {"role": "user", "content": "Remember the word cache."},
                {"role": "assistant", "content": "I will remember cache."},
                {"role": "user", "content": "Which word did I ask you to remember?"},
            ]
        rendered = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = tok(rendered, add_special_tokens=False).input_ids
        direct = tok.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        assert ids == direct, f"Two template paths differ for case {index}"
        assert 0 < len(ids) <= 2016, (index, len(ids))
        rows.append(
            {
                "id": f"p{index:02d}",
                "messages": messages,
                "rendered": rendered,
                "prompt_token_ids": ids,
            }
        )

    path = root / "prompts.jsonl"
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    synthetic = tok(
        "The cache stores previous keys and values. " * 128, add_special_tokens=False
    ).input_ids[:128]
    assert len(synthetic) == 128
    (root / "bench-input.jsonl").write_text(
        json.dumps(
            {
                "id": "isl128",
                "prompt_token_ids": synthetic,
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    (root / "tokenizer-info.json").write_text(
        json.dumps(
            {
                "class": type(tok).__name__,
                "eos_token_id": tok.eos_token_id,
                "bos_token_id": tok.bos_token_id,
                "chat_template": tok.chat_template,
                "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    for row in rows:
        print(row["id"], len(row["prompt_token_ids"]), repr(row["rendered"][:100]))


if __name__ == "__main__":
    main()
