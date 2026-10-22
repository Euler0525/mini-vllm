"""Run controlled inference and record outputs, timings and provenance."""

import argparse
import hashlib
import importlib.metadata as metadata
import json
import subprocess
import sys
import time
from pathlib import Path


def git_commit(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def main() -> None:
    project = Path(__file__).resolve().parents[2]
    artifacts = project / "artifacts/inference"
    parser = argparse.ArgumentParser(
        description="Run single-request inference with HF, mini_llm or vLLM."
    )
    parser.add_argument(
        "--backend", choices=["hf", "mini", "vllm"], required=True
    )
    parser.add_argument(
        "--model", type=Path, default=project / "artifacts/models/qwen2.5-0.5b"
    )
    parser.add_argument("--inputs", type=Path, default=Path("prompts.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--logprobs", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if not args.model.is_absolute():
        args.model = project / args.model
    if not args.inputs.is_absolute():
        args.inputs = artifacts / "inputs" / args.inputs
    if not args.output.is_absolute():
        args.output = artifacts / "results" / args.output
    if min(args.max_new_tokens, args.max_model_len, args.repeats) <= 0:
        parser.error(
            "Token counts, context length and repeats must be positive"
        )
    if args.warmups < 0 or (args.limit is not None and args.limit <= 0):
        parser.error("Warmups must be non-negative and limit must be positive")
    raw = args.inputs.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Expected non-empty inputs with unique IDs")

    import torch
    from transformers import AutoTokenizer

    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.manual_seed(0)
    model_path = args.model.resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True
    )
    eos = tokenizer.eos_token_id
    for row in rows:
        ids = row["prompt_token_ids"]
        if not ids or any(
            type(t) is not int or t < 0 or t >= len(tokenizer) for t in ids
        ):
            raise ValueError(f"Invalid token IDs: {row['id']}")
        if len(ids) + args.max_new_tokens > args.max_model_len:
            raise ValueError(
                f"Input + output budget exceeds context: {row['id']}"
            )

    started = time.perf_counter()
    if args.backend == "hf":
        from transformers import AutoModelForCausalLM

        model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                dtype=torch.bfloat16,
                attn_implementation="eager",
            )
            .to("cuda")
            .eval()
        )
    elif args.backend == "mini":
        sys.path.insert(0, str(project))
        from mini_llm.config.model import ModelConfig
        from mini_llm.engine.kv_cache import KVCache
        from mini_llm.models import Qwen2p5ForCausalLM
        from mini_llm.utils import load_safetensors_weights

        config = ModelConfig.from_json_file(model_path / "config.json")
        model = Qwen2p5ForCausalLM(config)
        model.to(device="cuda", dtype=torch.bfloat16).eval()
        load_safetensors_weights(model, model_path / "model.safetensors")
    else:
        from vllm import LLM, SamplingParams

        model = LLM(
            model=str(model_path),
            dtype="bfloat16",
            seed=0,
            max_model_len=args.max_model_len,
            max_num_seqs=1,
            max_num_batched_tokens=1024,
            gpu_memory_utilization=0.70,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            enforce_eager=True,
            generation_config="vllm",
        )
        sampling = SamplingParams(
            temperature=0,
            top_p=1.0,
            top_k=-1,
            repetition_penalty=1.0,
            seed=0,
            max_tokens=args.max_new_tokens,
            ignore_eos=args.ignore_eos,
            logprobs=5 if args.logprobs else None,
        )
    if args.backend != "vllm":
        torch.cuda.synchronize()
    load_s = time.perf_counter() - started

    @torch.inference_mode()
    def infer(ids):
        if args.backend == "vllm":
            result = model.generate(
                [{"prompt_token_ids": ids}],
                sampling,
                use_tqdm=False,
            )[0]
            assert result.prompt_token_ids == ids
            output = result.outputs[0]
            top5 = []
            if args.logprobs and output.logprobs:
                values = sorted(
                    output.logprobs[0].items(),
                    key=lambda item: item[1].logprob,
                    reverse=True,
                )
                top5 = [
                    {"token_id": int(t), "logprob": float(v.logprob)}
                    for t, v in values[:5]
                ]
            return list(output.token_ids), top5, output.finish_reason

        x = torch.tensor([ids], device="cuda", dtype=torch.long)
        state = None
        if args.backend == "mini":
            state = KVCache.allocate(
                config,
                batch_size=1,
                max_model_len=args.max_model_len,
                device="cuda",
                dtype=torch.bfloat16,
            )
        generated, top5 = [], []
        reason = "max_tokens"
        for step in range(args.max_new_tokens):
            if args.backend == "hf":
                output = model(
                    input_ids=x, past_key_values=state, use_cache=True
                )
                logits = output.logits[:, -1]
                state = output.past_key_values
            else:
                logits = (
                    model.prefill(x, state)
                    if step == 0
                    else model.decode(x, state)
                )
            if step == 0 and args.logprobs:
                values, tokens = logits.float().log_softmax(-1).topk(5)
                top5 = [
                    {"token_id": int(t), "logprob": float(v)}
                    for t, v in zip(tokens[0].tolist(), values[0].tolist())
                ]
            token = int(logits.argmax(-1).item())
            generated.append(token)
            if token == eos and not args.ignore_eos:
                reason = "eos"
                break
            if step + 1 < args.max_new_tokens:
                x = torch.tensor([[token]], device="cuda", dtype=torch.long)
        return generated, top5, reason

    for _ in range(args.warmups):
        infer(rows[0]["prompt_token_ids"])
    if args.backend != "vllm":
        torch.cuda.synchronize()
    requests = []
    window_start = time.perf_counter()
    for repeat in range(args.repeats):
        for row in rows:
            if args.backend != "vllm":
                torch.cuda.synchronize()
            begin = time.perf_counter()
            ids, top5, reason = infer(row["prompt_token_ids"])
            if args.backend != "vllm":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - begin) * 1000
            requests.append(
                {
                    "id": row["id"],
                    "repeat": repeat,
                    "prompt_token_ids": row["prompt_token_ids"],
                    "token_ids": ids,
                    "offline_call_ms": elapsed_ms,
                    "first_top5": top5,
                    "finish_reason": reason,
                }
            )
    window_s = time.perf_counter() - window_start
    for request in requests:
        request["text"] = tokenizer.decode(
            request["token_ids"], skip_special_tokens=True
        )
    result = {
        "backend": args.backend,
        "model": str(model_path),
        "dtype": "bfloat16",
        "inputs_sha256": hashlib.sha256(raw).hexdigest(),
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "ignore_eos": args.ignore_eos,
        "logprobs": args.logprobs,
        "limit": args.limit,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "sampling": "greedy; repetition_penalty=1",
        "eos_token_id": eos,
        "model_load_s": load_s,
        "measurement_window_s": window_s,
        "environment": {
            "python": sys.executable,
            "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "transformers": metadata.version("transformers"),
            "vllm": metadata.version("vllm")
            if args.backend == "vllm"
            else None,
            "mini_llm_source_commit": git_commit(project),
            "vllm_source_commit": git_commit(project.parent / "vllm"),
            "vllm_import_path": (
                str(Path(sys.modules["vllm"].__file__).resolve())
                if args.backend == "vllm"
                else None
            ),
        },
        "requests": requests,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(f"Saved {len(requests)} requests to {args.output}")


if __name__ == "__main__":
    main()
