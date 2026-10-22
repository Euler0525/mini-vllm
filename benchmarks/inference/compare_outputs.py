"""Compare generated token sequences under matching inference configurations."""

import argparse
import json
from pathlib import Path


def index_requests(document):
    result = {}
    for row in document["requests"]:
        key = (row["id"], row["repeat"])
        if key in result:
            raise ValueError(f"Duplicate request: {key}")
        if (
            document["ignore_eos"]
            and len(row["token_ids"]) != document["max_new_tokens"]
        ):
            raise ValueError(f"Unexpected fixed output length: {key}")
        result[key] = row
    if not result:
        raise ValueError("No requests to compare")
    return result


def first_difference(a, b):
    for position, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return position
    return None if len(a) == len(b) else min(len(a), len(b))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="Reference result JSON")
    parser.add_argument(
        "candidates", type=Path, nargs="+", help="Candidate result JSONs"
    )
    args = parser.parse_args()
    results = Path(__file__).resolve().parents[2] / "artifacts/inference/results"
    paths = [
        p if p.is_absolute() else results / p
        for p in [args.reference, *args.candidates]
    ]
    documents = [json.loads(p.read_text()) for p in paths]
    reference = documents[0]
    gold = index_requests(reference)
    report = []
    for document in documents[1:]:
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
            if document[field] != reference[field]:
                raise ValueError(f"Experiment configuration differs: {field}")
        candidate = index_requests(document)
        if gold.keys() != candidate.keys():
            raise ValueError("Missing or extra requests")
        details = []
        for key, a in gold.items():
            b = candidate[key]
            if a["prompt_token_ids"] != b["prompt_token_ids"]:
                raise ValueError(f"Prompt token IDs differ: {key}")
            position = first_difference(a["token_ids"], b["token_ids"])
            if position is not None:
                details.append(
                    {
                        "id": key[0],
                        "repeat": key[1],
                        "first_difference_index": position,
                        "reference_token": (
                            a["token_ids"][position]
                            if position < len(a["token_ids"])
                            else None
                        ),
                        "candidate_token": (
                            b["token_ids"][position]
                            if position < len(b["token_ids"])
                            else None
                        ),
                    }
                )
        report.append(
            {
                "reference": reference["backend"],
                "candidate": document["backend"],
                "total": len(gold),
                "identical": len(gold) - len(details),
                "differences": details,
            }
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
