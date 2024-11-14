from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from typing import List, Union
import torch
import logging

from logging import getLogger

logger = getLogger()


# Get the output from the model.
# Note: This method is copied from the source given below:
# https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/6-PRUNING/6_3_pruning_structured_llama3.2-1b_OK.ipynb
def get_output(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 50,
    device: str = "cuda",
    apply_chat_template: bool = False,
    quiet: bool = False,
) -> List[Union[str, torch.Tensor]]:
    """
    Get the output from the model.

    Args:
    - prompt: Prompt to generate the output.
    - model: Model to use.
    - tokenizer: Tokenizer to use.
    - max_new_tokens: Maximum number of tokens to generate.
    - device: Device to use.
    - apply_chat_template: Apply chat template to the model
    - quiet: If True, the output will not be printed.

    Returns:
    - generated: Generated output.
    - logits: Logits generated.
    """

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if apply_chat_template:
        prompt = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    streamer = TextStreamer(tokenizer, skip_prompt=False)
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        temperature=None,
        top_p=None,
        use_cache=True,
        streamer=streamer if quiet is False else None,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,
        no_repeat_ngram_size=None,
    )

    with torch.no_grad():
        logits = model(**inputs).logits.cpu()
    generated = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Ensure the logger has a FileHandler
    file_handler = None
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            file_handler = handler
            break

    if file_handler:
        file_handler.stream.write("\n\n" + generated + "\n\n")
        file_handler.flush()

    print()

    return generated, logits
