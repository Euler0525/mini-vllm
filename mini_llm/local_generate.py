import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from mini_llm.config.model import ModelConfig
from mini_llm.config.sampling import SamplingParams
from mini_llm.config.settings import (
    DEFAULT_MAX_NEW_TOKENS,
    REFERENCE_PROMPT,
    ModelSettings,
)
from mini_llm.engine.generation import generate_tokens
from mini_llm.engine.kv_cache import KVCache
from mini_llm.models import Qwen2p5ForCausalLM
from mini_llm.utils import load_safetensors_weights


@dataclass(frozen=True)
class TextGenerationResult:
    prompt_tokens: int
    generated_token_ids: list[int]
    generated_text: str
    stop_reason: str


def generate(
    prompt: str,
    sampling_params: SamplingParams,
    max_model_len: int,
    settings: ModelSettings | None = None,
) -> TextGenerationResult:
    settings = settings or ModelSettings()
    model_path = Path(snapshot_download(
        settings.model_id,
        revision=settings.revision,
        local_files_only=settings.local_files_only,
    ))
    config = ModelConfig.from_json_file(model_path / "config.json")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )
    rendered_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = tokenizer(
        rendered_prompt,
        return_tensors="pt",
    ).input_ids.to(settings.device)
    print("mini_vllm input IDs: ", input_ids.tolist())

    model = Qwen2p5ForCausalLM(config).to(
        device=settings.device,
        dtype=settings.dtype,
    ).eval()
    load_safetensors_weights(model, model_path / "model.safetensors")
    kv_cache = KVCache.allocate(
        config,
        batch_size=1,
        max_model_len=max_model_len,
        device=settings.device,
        dtype=settings.dtype,
    )
    output = generate_tokens(
        model,
        input_ids,
        kv_cache,
        sampling_params,
        tokenizer.eos_token_id,
    )
    return TextGenerationResult(
        prompt_tokens=input_ids.shape[1],
        generated_token_ids=output.token_ids,
        generated_text=tokenizer.decode(
            output.token_ids,
            skip_special_tokens=True,
        ),
        stop_reason=output.stop_reason,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the custom Qwen2.5 model for one text request."
    )
    parser.add_argument("--prompt", default=REFERENCE_PROMPT)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
    )
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ignore-eos", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = generate(
        args.prompt,
        SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
            ignore_eos=args.ignore_eos,
            seed=args.seed,
        ),
        args.max_model_len,
    )
    print(f"prompt tokens: {result.prompt_tokens}")
    print(f"generated token ids: {result.generated_token_ids}")
    print(f"stop reason: {result.stop_reason}")
    print(f"\n{result.generated_text}")

    from mini_llm.reference import load_tokenizer
    print("HF token IDs:", result.generated_token_ids)
    tokenizer = load_tokenizer(ModelSettings())
    for index, token_id in enumerate(result.generated_token_ids):
        print(index, token_id, repr(tokenizer.decode([token_id])))
        if index == 20:
            break


if __name__ == "__main__":
    main()
