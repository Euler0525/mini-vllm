"""Audit saved divergence traces, interventions, and the first RMSNorm rounding."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open


def read(path):
    return json.loads(path.read_text())


def requests(path):
    return {r["id"]: r for r in read(path)["requests"]}


def logits(path, case):
    state = torch.load(path / f"{case}.pt", weights_only=True)
    value = state["logits"]
    return value.reshape(-1, value.shape[-1])[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root
    output = args.output or root / "verification.json"
    if output.exists():
        raise FileExistsError(output)
    project = Path(__file__).resolve().parents[2]
    source = project / "artifacts/inference/results"
    original_hf = read(source / "hf-fixed16.json")
    original = {
        name: requests(source / f"{name}-fixed16.json")
        for name in ("hf", "mini", "vllm")
    }
    hf_path = root / "hf-replay-bfloat16-baseline"
    hf = requests(hf_path / "result.json")
    summary = {"cases": len(hf), "rms_audit_device": args.device}

    manifest = read(root / "source-manifest.json")
    model_path = Path(original_hf["model"])
    checks = {
        name: hashlib.sha256((model_path / name).read_bytes()).hexdigest() == digest
        for name, digest in manifest["model_files"].items()
    }
    checks.update(
        {
            f"{name}-fixed16.json": hashlib.sha256(
                (source / f"{name}-fixed16.json").read_bytes()
            ).hexdigest()
            == manifest[name]["sha256"]
            for name in ("hf", "mini", "vllm")
        }
    )
    assert all(checks.values())
    summary["original_files_unchanged"] = checks

    replay = {}
    for backend in ("hf", "mini", "vllm"):
        path = root / f"{backend}-replay-bfloat16-baseline/result.json"
        document = read(path)
        rows = {r["id"]: r for r in document["requests"]}
        matches = [
            name
            for name, row in rows.items()
            if row["token_ids"] == original[backend][name]["token_ids"]
        ]
        assert len(matches) == len(hf)
        assert not document["weights"]["mismatches"]
        replay[backend] = {
            "identical_original_sequences": len(matches),
            "weights": document["weights"],
        }
    summary["baseline_replay"] = replay

    controlled = root / "torch214/hf-replay-bfloat16-baseline"
    control = requests(controlled / "result.json")
    summary["environment_control"] = {
        "identical_sequences": sum(
            control[name]["token_ids"] == row["token_ids"] for name, row in hf.items()
        ),
        "identical_divergence_logits": sum(
            torch.equal(logits(controlled, name), logits(hf_path, name)) for name in hf
        ),
    }
    reference_path = root / "vllm-replay-bfloat16-all"
    reference = requests(reference_path / "result.json")
    summary["reference_replay"] = {
        "identical_sequences": sum(
            reference[name]["token_ids"] == row["token_ids"] for name, row in hf.items()
        ),
        "identical_tokens": sum(
            x == y
            for name, row in hf.items()
            for x, y in zip(row["token_ids"], reference[name]["token_ids"])
        ),
    }
    assert summary["reference_replay"]["identical_sequences"] == len(hf)

    kv_path = root / "vllm-replay-bfloat16-kv-audit"
    kv_rows = requests(kv_path / "result.json")
    kv_checks, attention_differences, max_attention_difference = 0, 0, 0.0
    for name in hf:
        assert kv_rows[name]["token_ids"] == original["vllm"][name]["token_ids"]
        states = torch.load(kv_path / f"{name}.pt", weights_only=True)
        assert states["kv_checks"] == 16 * 24 * 2
        kv_checks += states["kv_checks"]
        assert len(states["attention_checks"]) == 24
        for item in states["attention_checks"].values():
            assert item["same_paged_kv"]
            difference = item["max_abs_eager_vs_flash"]
            attention_differences += difference > 0
            max_attention_difference = max(max_attention_difference, difference)
    summary["paged_kv_audit"] = {
        "exact_key_or_value_history_checks": kv_checks,
        "checks_with_mismatches": 0,
        "original_sequences_preserved": len(kv_rows),
        "same_qkv_attention_outputs_differing": attention_differences,
        "same_qkv_attention_comparisons": len(hf) * 24,
        "max_attention_absolute_difference": max_attention_difference,
    }

    four_path = root / "vllm-replay-bfloat16-norm+rope+split+attention"
    four = requests(four_path / "result.json")
    summary["four_component_reference_replay"] = {
        "identical_sequences": sum(
            four[name]["token_ids"] == row["token_ids"] for name, row in hf.items()
        ),
        "identical_tokens": sum(
            x == y
            for name, row in hf.items()
            for x, y in zip(row["token_ids"], four[name]["token_ids"])
        ),
    }
    assert summary["four_component_reference_replay"]["identical_sequences"] == len(hf)

    components = {}
    for file in sorted(root.rglob("vllm-forced-bfloat16-*/result.json")):
        variant = file.parent.name.removeprefix("vllm-forced-bfloat16-")
        if variant in components:
            raise ValueError(f"Duplicate variant: {variant}")
        doc = read(file)
        rows = {r["id"]: r for r in doc["requests"]}
        assert rows.keys() == hf.keys()
        for name, row in rows.items():
            k = hf[name]["k"]
            assert row["token_ids"][:k] == hf[name]["token_ids"][:k]
        components[variant] = {
            "result_path": str(file),
            "identical_argmax": [
                name
                for name in hf
                if rows[name]["token_ids"][-1] == hf[name]["token_ids"][hf[name]["k"]]
            ],
            "identical_full_vocab_logits": [
                name
                for name in hf
                if torch.equal(logits(file.parent, name), logits(hf_path, name))
            ],
        }
    assert len(components["all"]["identical_full_vocab_logits"]) == len(hf)
    assert len(
        components["norm+rope+split+attention"]["identical_full_vocab_logits"]
    ) == len(hf)
    summary["component_interventions"] = components

    with safe_open(str(model_path / "model.safetensors"), framework="pt") as f:
        weight = f.get_tensor("model.layers.0.input_layernorm.weight").to(
            device=args.device, dtype=torch.bfloat16
        )
    rms, cases = [], []
    vllm_path = root / "vllm-replay-bfloat16-baseline"
    vllm = requests(vllm_path / "result.json")
    for name, row in hf.items():
        a = torch.load(hf_path / f"{name}.pt", weights_only=True)
        b = torch.load(vllm_path / f"{name}.pt", weights_only=True)
        key = "model.layers.0.input_layernorm"
        x = a[key + ":in"][0].reshape(-1, 896)
        other_x = b[key + ":in"][0].reshape(-1, 896)
        assert torch.equal(x, other_x)
        expected = a[key + ":out"].reshape(-1, 896)
        candidate = b[key + ":out"].reshape(-1, 896)
        value = x.to(args.device).float()
        normalized = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
        twice = (normalized.to(torch.bfloat16) * weight).cpu()
        once = (normalized * weight.float()).to(torch.bfloat16).cpu()
        rms.append(
            {
                "id": name,
                "identical_inputs": True,
                "elements": x.numel(),
                "hf_vs_vllm_unequal": int((expected != candidate).sum()),
                "hf_vs_double_round_unequal": int((expected != twice).sum()),
                "vllm_vs_single_round_unequal": int((candidate != once).sum()),
                "max_absolute_difference": float(
                    (expected.float() - candidate.float()).abs().max()
                ),
            }
        )
        k = row["k"]
        ha, va = row["scores"][k], vllm[name]["scores"][k]
        ht, vt = (
            original["hf"][name]["token_ids"][k],
            original["vllm"][name]["token_ids"][k],
        )

        def margin(scores):
            return (
                scores["candidates"][str(ht)]["logit"]
                - scores["candidates"][str(vt)]["logit"]
            )

        eos = original_hf["eos_token_id"]
        before = row["token_ids"][:k]
        cases.append(
            {
                "id": name,
                "k": k,
                "hf_token": ht,
                "vllm_token": vt,
                "hf_candidate_margin": margin(ha),
                "vllm_candidate_margin": margin(va),
                "after_eos": eos in before,
                "single_interventions_restoring_hf": [
                    variant
                    for variant in ("norm", "rope", "activation", "split", "attention")
                    if name in components[variant]["identical_argmax"]
                ],
                "reference_full_logits_equal": name
                in components["all"]["identical_full_vocab_logits"],
            }
        )
    summary["first_rms_norm"] = rms
    summary["per_case"] = cases
    with output.open("x") as file:
        json.dump(summary, file, indent=2)
    print("Audited", len(cases), "divergences,", len(components), "interventions.")
    print("Environment control:", summary["environment_control"])
    print("Reference replay:", summary["reference_replay"])
    print("Saved", output)


if __name__ == "__main__":
    main()
