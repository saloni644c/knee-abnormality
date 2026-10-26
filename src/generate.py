```python
import torch

from .prompt import build_messages
from .config import MAX_NEW_TOKENS


@torch.inference_mode()
def generate_one(report_text, model, processor, max_new_tokens=MAX_NEW_TOKENS):
    inputs = processor.apply_chat_template(
        build_messages(report_text),
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    n_in = inputs["input_ids"].shape[-1]

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    return processor.decode(
        out[0][n_in:],
        skip_special_tokens=True,
    )


@torch.inference_mode()
def generate_batch(
    texts,
    model,
    processor,
    max_new_tokens=MAX_NEW_TOKENS,
):
    inputs = processor.apply_chat_template(
        [build_messages(text) for text in texts],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    input_length = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    return processor.batch_decode(
        outputs[:, input_length:],
        skip_special_tokens=True,
    )
```