import torch

from typing import Any
from mini_llm.engine.state import DecodeState, StepOutput
from mini_llm.utils import timed_device_call


def prefill(
    model: Any,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> StepOutput:
    """Process the complete prompt once and create the initial KV cache."""
    with torch.inference_mode():
        outputs, elapsed_ms = timed_device_call(
            lambda: model(
                input_ids=token_ids,
                attention_mask=attention_mask,
                use_cache=True,
            ),
            token_ids.device,
        )

    last_logits = outputs.logits[:, -1, :]
    return StepOutput(
        next_token_id=last_logits.argmax(dim=-1).item(),
        state=DecodeState(
            past_key_values=outputs.past_key_values,
            attention_mask=attention_mask,
        ),
        elapsed_ms=elapsed_ms,
        input_shape=list(token_ids.shape),
        logits_shape=list(outputs.logits.shape),
    )


def decode(
    model: Any,
    token_id: int,
    state: DecodeState,
) -> StepOutput:
    """Append one token, reuse the KV cache, and predict one new token."""
    device = state.attention_mask.device
    token_ids = torch.tensor([[token_id]], dtype=torch.long, device=device)
    attention_mask = torch.cat(
        [
            state.attention_mask,
            torch.ones(
                (state.attention_mask.shape[0], 1),
                dtype=state.attention_mask.dtype,
                device=device,
            ),
        ],
        dim=-1,
    )

    with torch.inference_mode():
        outputs, elapsed_ms = timed_device_call(
            lambda: model(
                input_ids=token_ids,
                attention_mask=attention_mask,
                past_key_values=state.past_key_values,
                use_cache=True,
            ),
            device,
        )

    last_logits = outputs.logits[:, -1, :]
    # if attention_mask.shape[1] == 53:
    #     scores = last_logits[0].float()
    #     values, indices = scores.topk(5)
    #     print("HF logits dtype:", last_logits.dtype)
    #     print("HF plans  [6649]:", scores[6649].item())
    #     print("HF choices[11454]:", scores[11454].item())
    #     print("HF top5 (token_id, logit):",
    #           list(zip(indices.tolist(), values.tolist())))
    return StepOutput(
        next_token_id=last_logits.argmax(dim=-1).item(),
        state=DecodeState(
            past_key_values=outputs.past_key_values,
            attention_mask=attention_mask,
        ),
        elapsed_ms=elapsed_ms,
        input_shape=list(token_ids.shape),
        logits_shape=list(outputs.logits.shape),
    )
