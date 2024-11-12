from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# Get the output from the model.
# Note: This method is copied from the source given below:
# https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/6-PRUNING/6_3_pruning_structured_llama3.2-1b_OK.ipynb
def get_output(prompt: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, max_new_tokens: int = 50, device: str = 'cuda', apply_chat_template: bool = False) -> str:
    """
    Get the output from the model.
    
    Args:
    - prompt: Prompt to generate the output.
    - model: Model to use.
    - tokenizer: Tokenizer to use.
    - max_new_tokens: Maximum number of tokens to generate.
    - device: Device to use.
    - apply_chat_template: Apply chat template to the model

    Returns:
    - generated: Generated output.
    """

    if apply_chat_template:
        prompt = [{'role': 'user', 'content': prompt}]
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    streamer = TextStreamer(tokenizer, skip_prompt=False)
    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        temperature=None,
        top_p=None,
        use_cache=True,
        streamer=streamer,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,          # Disable sampling
        num_beams=1,              # Use beam search
        early_stopping=True,      # Stop when end-of-sequence token is generated
        no_repeat_ngram_size=None    # Prevent repetition of 2-grams
    )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated