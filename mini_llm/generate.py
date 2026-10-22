import argparse

from mini_llm.config import (
    DEFAULT_MAX_NEW_TOKENS,
    REFERENCE_PROMPT,
    ModelSettings,
)
from mini_llm.engine import GenerationResult, decode, prefill
from mini_llm.reference import encode_prompt, load_model, load_tokenizer


def generate(
    prompt: str,
    max_new_tokens: int,
    settings: ModelSettings | None = None,
) -> GenerationResult:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    settings = settings or ModelSettings()
    tokenizer = load_tokenizer(settings)
    model = load_model(settings, attention_implementation="eager")
    device = str(next(model.parameters()).device)
    inputs = encode_prompt(tokenizer, prompt, device)
    print("HF input IDs: ", inputs.input_ids.tolist())

    output = prefill(model, inputs.input_ids, inputs.attention_mask)
    token_ids = [output.next_token_id]
    state = output.state
    prefill_ms = output.elapsed_ms
    decode_ms = 0.0
    decode_calls = 0

    while token_ids[-1] != tokenizer.eos_token_id and len(token_ids) < max_new_tokens:
        output = decode(model, token_ids[-1], state)
        token_ids.append(output.next_token_id)
        state = output.state
        decode_ms += output.elapsed_ms
        decode_calls += 1

    stop_reason = (
        "eos" if token_ids[-1] == tokenizer.eos_token_id else "max_new_tokens"
    )
    return GenerationResult(
        prompt_tokens=inputs.input_ids.shape[-1],
        generated_token_ids=token_ids,
        generated_text=tokenizer.decode(token_ids, skip_special_tokens=True),
        stop_reason=stop_reason,
        prefill_calls=1,
        decode_calls=decode_calls,
        prefill_ms=prefill_ms,
        decode_ms=decode_ms,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run greedy generation with explicit prefill and decode."
    )
    parser.add_argument("--prompt", default=REFERENCE_PROMPT)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = generate(args.prompt, args.max_new_tokens)
    print(f"prompt tokens: {result.prompt_tokens}")
    print(f"generated tokens: {len(result.generated_token_ids)}")
    print(f"generated token ids: {result.generated_token_ids}")
    print(f"prefill calls: {result.prefill_calls}")
    print(f"decode calls: {result.decode_calls}")
    print(f"prefill: {result.prefill_ms:.3f} ms")
    print(f"average decode: {result.average_decode_ms:.3f} ms")
    print(f"stop reason: {result.stop_reason}")
    print(f"\n{result.generated_text}")

    print("HF token IDs:", result.generated_token_ids)
    tokenizer = load_tokenizer(ModelSettings())
    for index, token_id in enumerate(result.generated_token_ids):
        print(index, token_id, repr(tokenizer.decode([token_id])))
        if index == 20:
            break


if __name__ == "__main__":
    main()
