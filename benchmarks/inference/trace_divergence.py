"""Replay divergent requests and run controlled, teacher-forced numeric probes.

Diagnostic interventions are confined to the loaded model in this process.
The checkpoint, production runtime, and original benchmark results are unchanged.
"""

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path


def describe_logits(logits, candidates):
    """Describe raw scores before any diagnostic token forcing."""
    vector = logits.detach().reshape(-1, logits.shape[-1])[-1]
    logprobs = vector.float().log_softmax(-1)
    values, tokens = logprobs.topk(10)
    return {
        "argmax": int(vector.argmax().item()),
        "dtype": str(vector.dtype),
        "top10": [
            {
                "token_id": token,
                "logprob": value,
                "logit": float(vector[token].item()),
            }
            for token, value in zip(tokens.tolist(), values.tolist())
        ],
        "candidates": {
            str(token): {
                "logit": float(vector[token].item()),
                "logprob": float(logprobs[token].item()),
            }
            for token in candidates
        },
    }


def cpu_tree(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, (tuple, list)):
        return tuple(cpu_tree(v) for v in value)
    return None


def install_hooks(model, context):
    suffixes = (
        "embed_tokens",
        "input_layernorm",
        "post_attention_layernorm",
        "q_proj",
        "k_proj",
        "v_proj",
        "qkv_proj",
        "rotary_emb",
        "attn",
        "o_proj",
        "gate_proj",
        "up_proj",
        "gate_up_proj",
        "act_fn",
        "down_proj",
        "norm",
        "lm_head",
    )

    def before(name):
        def capture(module, inputs):
            state = context()
            if state["active"] and state["step"] == state["target"]:
                state["states"][name + ":in"] = cpu_tree(inputs)

        return capture

    def after(name):
        def capture(module, inputs, output):
            state = context()
            if state["active"] and state["step"] == state["target"]:
                # The full vocabulary projection is recorded separately.
                if name == "lm_head":
                    return
                state["states"][name + ":out"] = cpu_tree(output)

        return capture

    for name, module in model.named_modules():
        if name.endswith(suffixes):
            module.register_forward_pre_hook(before(name))
            module.register_forward_hook(after(name))


def verify_weights(model, model_path):
    """Check packed and unpacked learned parameters against the same snapshot."""
    import torch
    from safetensors import safe_open

    checked, mismatches = [], []
    with safe_open(str(Path(model_path) / "model.safetensors"), framework="pt") as f:
        for name, parameter in model.named_parameters():
            if ".qkv_proj." in name:
                expected = torch.cat(
                    [
                        f.get_tensor(name.replace(".qkv_proj.", f".{part}_proj."))
                        for part in ("q", "k", "v")
                    ]
                )
            elif ".gate_up_proj." in name:
                expected = torch.cat(
                    [
                        f.get_tensor(name.replace(".gate_up_proj.", f".{part}_proj."))
                        for part in ("gate", "up")
                    ]
                )
            else:
                source = (
                    "model.embed_tokens.weight" if name == "lm_head.weight" else name
                )
                expected = f.get_tensor(source)
            actual = parameter.detach().cpu()
            expected = expected.to(actual.dtype)
            if actual.shape != expected.shape or not torch.equal(actual, expected):
                mismatches.append(name)
            checked.append(name)
    return {"checked_parameters": len(checked), "mismatches": mismatches}


def configure_worker(model, variant):
    """Install reversible ablations inside a vLLM worker, never in its checkout."""
    import types

    import torch
    import torch.nn.functional as functional

    if not hasattr(model, "_diag_originals"):
        model._diag_originals = {
            name: module.forward for name, module in model.named_modules()
        }
        model._diag_original_compute = model.compute_logits
        model._diag = {"active": False}
        install_hooks(model, lambda: model._diag)

        def compute(hidden_states):
            logits = model._diag_original_compute(hidden_states)
            state = model._diag
            if not state["active"]:
                return logits
            state["scores"].append(describe_logits(logits, state["candidates"]))
            step = state["step"]
            if step == state["target"]:
                state["states"]["final_hidden"] = cpu_tree(hidden_states)
                state["states"]["logits"] = cpu_tree(logits)
            state["step"] += 1
            if step < len(state["force_ids"]):
                logits = torch.full_like(logits, float("-inf"))
                logits[..., state["force_ids"][step]] = 0
            return logits

        model.compute_logits = compute

    for name, module in model.named_modules():
        module.forward = model._diag_originals[name]

    selected = set(variant.split("+"))
    if variant == "all":
        selected = {"norm", "rope", "activation", "split", "attention"}
    changes = []

    def reference_norm(self, x, residual=None):
        # HF materializes the BF16 residual sum before normalizing it.
        if residual is not None:
            x = x + residual
            residual = x
        original_dtype = x.dtype
        value = x.float()
        variance = value.square().mean(dim=-1, keepdim=True)
        normalized = (value * torch.rsqrt(variance + self.variance_epsilon)).to(
            original_dtype
        )
        normalized = self.weight * normalized
        return normalized if residual is None else (normalized, residual)

    def split_projection(sizes):
        def forward(self, x):
            weights = self.weight.split(sizes, dim=0)
            biases = (
                self.bias.split(sizes) if self.bias is not None else [None] * len(sizes)
            )
            return torch.cat(
                [functional.linear(x, w, b) for w, b in zip(weights, biases)], dim=-1
            ), None

        return forward

    def reference_activation(self, x):
        gate, up = x.chunk(2, dim=-1)
        return functional.silu(gate) * up

    def reference_attention(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        count = positions.numel()
        q = q.view(count, self.num_heads, self.head_dim).transpose(0, 1).unsqueeze(0)
        k = k.view(count, self.num_kv_heads, self.head_dim).transpose(0, 1).unsqueeze(0)
        v = v.view(count, self.num_kv_heads, self.head_dim).transpose(0, 1).unsqueeze(0)
        previous = getattr(self, "_diag_kv", None)
        length = 0 if previous is None else previous[0].shape[2]
        assert torch.equal(
            positions, torch.arange(length, length + count, device=positions.device)
        )
        if previous is not None:
            k, v = (
                torch.cat((previous[0], k), dim=2),
                torch.cat((previous[1], v), dim=2),
            )
        self._diag_kv = (k, v)
        repeats = self.num_heads // self.num_kv_heads
        k, v = k.repeat_interleave(repeats, dim=1), v.repeat_interleave(repeats, dim=1)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        mask = torch.zeros((count, k.shape[2]), dtype=q.dtype, device=q.device)
        allowed = (
            torch.arange(k.shape[2], device=q.device)[None, :] <= positions[:, None]
        )
        mask.masked_fill_(~allowed, torch.finfo(q.dtype).min)
        scores = scores + mask
        probabilities = scores.softmax(dim=-1, dtype=torch.float32).to(q.dtype)
        output = torch.matmul(probabilities, v).transpose(1, 2).reshape(count, -1)
        output, _ = self.o_proj(output)
        return output

    def audit_attention(original):
        def forward(self, query, key, value, **kwargs):
            from vllm.model_executor.layers.attention.attention import (
                get_attention_context,
            )

            output = original(query, key, value, **kwargs)
            state = model._diag
            if not state["active"]:
                return output
            metadata, _, cache, _ = get_attention_context(self.layer_name)
            count = metadata.num_actual_tokens
            head_size = self.head_size
            current_k = key[:count].reshape(count, self.num_kv_heads, head_size).clone()
            current_v = (
                value[:count].reshape(count, self.num_kv_heads, head_size).clone()
            )
            previous = getattr(self, "_diag_kv", None)
            if previous is not None:
                current_k = torch.cat((previous[0], current_k))
                current_v = torch.cat((previous[1], current_v))
            self._diag_kv = (current_k, current_v)
            length = current_k.shape[0]
            assert int(metadata.seq_lens[0].item()) == length
            key_cache, value_cache = cache.transpose(1, 2).split(head_size, dim=-1)
            positions = torch.arange(length, device=query.device)
            block_size = key_cache.shape[1]
            blocks = metadata.block_table[0, positions // block_size].long()
            slots = positions % block_size
            paged_k, paged_v = key_cache[blocks, slots], value_cache[blocks, slots]
            assert torch.equal(paged_k, current_k), (
                self.layer_name,
                "key cache mismatch",
            )
            assert torch.equal(paged_v, current_v), (
                self.layer_name,
                "value cache mismatch",
            )
            state["kv_checks"] = state.get("kv_checks", 0) + 2
            if state["step"] == state["target"]:
                heads = query.shape[-1] // head_size
                q = query[:count].view(count, heads, head_size).transpose(0, 1)
                repeats = heads // self.num_kv_heads
                k = paged_k.transpose(0, 1).repeat_interleave(repeats, dim=0)
                v = paged_v.transpose(0, 1).repeat_interleave(repeats, dim=0)
                scores = torch.matmul(q, k.transpose(-2, -1)) * (head_size**-0.5)
                mask = torch.zeros(
                    (count, length), dtype=query.dtype, device=query.device
                )
                allowed = positions[None, :] <= positions[-count:, None]
                scores = scores + mask.masked_fill(
                    ~allowed, torch.finfo(query.dtype).min
                )
                probabilities = scores.softmax(-1, dtype=torch.float32).to(query.dtype)
                eager = (
                    torch.matmul(probabilities, v).transpose(0, 1).reshape(count, -1)
                )
                state.setdefault("attention_checks", {})[self.layer_name] = {
                    "same_paged_kv": True,
                    "max_abs_eager_vs_flash": float(
                        (eager.float() - output[:count].float()).abs().max().item()
                    ),
                }
            return output

        return forward

    for name, module in model.named_modules():
        replacement = None
        if "kv-audit" in selected and module.__class__.__name__ == "Attention":
            replacement = audit_attention(model._diag_originals[name])
        elif "norm" in selected and module.__class__.__name__ == "RMSNorm":
            replacement = reference_norm
        elif "rope" in selected and name.endswith("rotary_emb"):
            replacement = type(module).forward_native
        elif "activation" in selected and name.endswith("act_fn"):
            replacement = reference_activation
        elif "split" in selected and name.endswith("qkv_proj"):
            attention = model.get_submodule(name.rsplit(".", 1)[0])
            replacement = split_projection(
                [attention.q_size, attention.kv_size, attention.kv_size]
            )
        elif "split" in selected and name.endswith("gate_up_proj"):
            replacement = split_projection([module.weight.shape[0] // 2] * 2)
        elif "attention" in selected and name.endswith("self_attn"):
            replacement = reference_attention
        if replacement is not None:
            module.forward = types.MethodType(replacement, module)
            changes.append(name)
    return changes


def begin_worker(model, case, force):
    for module in model.modules():
        if hasattr(module, "_diag_kv"):
            module._diag_kv = None
    model._diag = {
        "active": True,
        "step": 0,
        "target": case["k"],
        "states": {},
        "scores": [],
        "candidates": [case["hf_token"], case["vllm_token"]],
        "force_ids": case["hf_ids"][: case["k"]] if force else [],
    }


def finish_worker(model, path):
    import torch

    state = model._diag
    state["active"] = False
    if "kv_checks" in state:
        state["states"]["kv_checks"] = state["kv_checks"]
        state["states"]["attention_checks"] = state.get("attention_checks", {})
    torch.save(state["states"], path)
    return state["scores"]


class DiagnosticWorker:
    """Expose diagnostics through named RPCs with ordinary JSON arguments."""

    def diag_verify_weights(self, model_path):
        return verify_weights(self.get_model(), model_path)

    def diag_configure(self, variant):
        return configure_worker(self.get_model(), variant)

    def diag_begin(self, case, force):
        begin_worker(self.get_model(), case, force)

    def diag_finish(self, path):
        return finish_worker(self.get_model(), path)


def run_vllm(args, cases, model_path):
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(model_path),
        dtype=args.dtype,
        seed=0,
        max_model_len=2048,
        max_num_seqs=1,
        max_num_batched_tokens=1024,
        gpu_memory_utilization=0.70,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        enforce_eager=True,
        generation_config="vllm",
        worker_extension_cls="trace_divergence.DiagnosticWorker",
    )
    weights = llm.collective_rpc("diag_verify_weights", args=(str(model_path),))[0]
    assert not weights["mismatches"], weights
    for variant in args.variants:
        target = args.output_dir / f"vllm-{args.mode}-{args.dtype}-{variant}"
        target.mkdir()
        changes = llm.collective_rpc("diag_configure", args=(variant,))[0]
        rows = []
        for case in cases:
            llm.collective_rpc("diag_begin", args=(case, args.mode == "forced"))
            count = case["k"] + 1 if args.mode == "forced" else 16
            params = SamplingParams(
                temperature=0,
                top_p=1.0,
                top_k=-1,
                repetition_penalty=1.0,
                seed=0,
                max_tokens=count,
                ignore_eos=True,
                logprobs=10,
            )
            result = llm.generate(
                [{"prompt_token_ids": case["prompt_ids"]}], params, use_tqdm=False
            )[0]
            assert result.prompt_token_ids == case["prompt_ids"]
            tokens = list(result.outputs[0].token_ids)
            if args.mode == "forced":
                assert tokens[: case["k"]] == case["hf_ids"][: case["k"]]
            scores = llm.collective_rpc(
                "diag_finish", args=(str(target / f"{case['id']}.pt"),)
            )[0]
            assert len(scores) == count
            rows.append(
                {
                    "id": case["id"],
                    "k": case["k"],
                    "token_ids": tokens,
                    "scores": scores,
                }
            )
            print(variant, case["id"], "at k:", tokens[case["k"]], flush=True)
        write_result(target, args, rows, weights, changes)


def run_torch(args, cases, model_path):
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    dtype = getattr(torch, args.dtype)
    if args.backend == "hf":
        from transformers import AutoModelForCausalLM

        model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                dtype=dtype,
                attn_implementation="eager",
            )
            .to("cuda")
            .eval()
        )
    else:
        from mini_llm.config.model import ModelConfig
        from mini_llm.engine.kv_cache import KVCache
        from mini_llm.models import Qwen2p5ForCausalLM
        from mini_llm.utils import load_safetensors_weights

        config = ModelConfig.from_json_file(model_path / "config.json")
        model = Qwen2p5ForCausalLM(config).to(device="cuda", dtype=dtype).eval()
        load_safetensors_weights(model, model_path / "model.safetensors")
    weights = verify_weights(model, model_path)
    assert not weights["mismatches"], weights
    target = args.output_dir / f"{args.backend}-{args.mode}-{args.dtype}-baseline"
    target.mkdir()
    context = {"active": False}
    install_hooks(model, lambda: context)
    rows = []
    with torch.inference_mode():
        for case in cases:
            context = {"active": True, "step": 0, "target": case["k"], "states": {}}
            state = (
                None
                if args.backend == "hf"
                else KVCache.allocate(
                    config, batch_size=1, max_model_len=2048, device="cuda", dtype=dtype
                )
            )
            x = torch.tensor([case["prompt_ids"]], device="cuda")
            tokens, scores = [], []
            count = case["k"] + 1 if args.mode == "forced" else 16
            for step in range(count):
                context["step"] = step
                if args.backend == "hf":
                    output = model(input_ids=x, past_key_values=state, use_cache=True)
                    logits, state = output.logits[:, -1], output.past_key_values
                else:
                    logits = (
                        model.prefill(x, state) if step == 0 else model.decode(x, state)
                    )
                scores.append(
                    describe_logits(logits, [case["hf_token"], case["vllm_token"]])
                )
                if step == case["k"]:
                    context["states"]["logits"] = cpu_tree(logits)
                token = scores[-1]["argmax"]
                if args.mode == "forced" and step < case["k"]:
                    token = case["hf_ids"][step]
                tokens.append(token)
                x = torch.tensor([[token]], device="cuda")
            context["active"] = False
            torch.save(context["states"], target / f"{case['id']}.pt")
            rows.append(
                {
                    "id": case["id"],
                    "k": case["k"],
                    "token_ids": tokens,
                    "scores": scores,
                }
            )
            print(args.backend, case["id"], "at k:", tokens[case["k"]], flush=True)
    write_result(target, args, rows, weights, [])


def write_result(target, args, rows, weights, changes):
    import torch

    result = {
        "backend": args.backend,
        "dtype": args.dtype,
        "mode": args.mode,
        "weights": weights,
        "intervened_modules": changes,
        "requests": rows,
        "environment": {
            "python": sys.executable,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "transformers": importlib.metadata.version("transformers"),
        },
    }
    (target / "result.json").write_text(json.dumps(result, indent=2))
    print("Saved", target / "result.json", flush=True)


def main():
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("hf", "mini", "vllm"), required=True)
    parser.add_argument("--mode", choices=("replay", "forced"), default="forced")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--variants", nargs="+", default=["baseline"])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.backend != "vllm" and args.variants != ["baseline"]:
        parser.error("Operator interventions are implemented only for vLLM")
    allowed = {
        "baseline",
        "norm",
        "rope",
        "activation",
        "split",
        "attention",
        "all",
        "kv-audit",
    }
    for variant in args.variants:
        if not set(variant.split("+")) <= allowed:
            parser.error(f"Unknown variant: {variant}")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = project / "artifacts/inference/results"
    hf = json.loads((root / "hf-fixed16.json").read_text())
    vl = json.loads((root / "vllm-fixed16.json").read_text())
    mini = json.loads((root / "mini-fixed16.json").read_text())
    for document in (mini, vl):
        for field in (
            "model",
            "dtype",
            "inputs_sha256",
            "max_new_tokens",
            "max_model_len",
            "ignore_eos",
            "logprobs",
            "limit",
            "repeats",
            "sampling",
            "eos_token_id",
        ):
            if document[field] != hf[field]:
                raise ValueError(f"Experiment configuration differs: {field}")
    if hf["repeats"] != 1 or hf["max_new_tokens"] != 16 or not hf["ignore_eos"]:
        raise ValueError(
            "This diagnostic requires one repeat, 16 tokens, and ignore_eos"
        )
    manifest_path = args.output_dir / "source-manifest.json"
    if not manifest_path.exists():
        manifest = {
            name: {
                "sha256": hashlib.sha256(
                    (root / f"{name}-fixed16.json").read_bytes()
                ).hexdigest(),
                "environment": document["environment"],
            }
            for name, document in (("hf", hf), ("mini", mini), ("vllm", vl))
        }
        manifest["model_files"] = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(hf["model"]).iterdir())
            if path.is_file()
        }
        with manifest_path.open("x") as file:
            json.dump(manifest, file, indent=2)
    other = {(r["id"], r["repeat"]): r for r in vl["requests"]}
    cases = []
    for a in hf["requests"]:
        b = other[(a["id"], a["repeat"])]
        assert a["prompt_token_ids"] == b["prompt_token_ids"]
        assert len(a["token_ids"]) == len(b["token_ids"]) == 16
        k = next(
            (
                i
                for i, (x, y) in enumerate(zip(a["token_ids"], b["token_ids"]))
                if x != y
            ),
            None,
        )
        if k is not None:
            cases.append(
                {
                    "id": a["id"],
                    "k": k,
                    "prompt_ids": a["prompt_token_ids"],
                    "hf_ids": a["token_ids"],
                    "hf_token": a["token_ids"][k],
                    "vllm_token": b["token_ids"][k],
                }
            )
    if not cases:
        print("No HF/vLLM token mismatch; skip diagnostics.")
        return
    for variant in args.variants:
        target = args.output_dir / f"{args.backend}-{args.mode}-{args.dtype}-{variant}"
        if target.exists():
            raise FileExistsError(f"Choose a new diagnostic output directory: {target}")
    runner = run_vllm if args.backend == "vllm" else run_torch
    runner(args, cases, Path(hf["model"]))


if __name__ == "__main__":
    main()
