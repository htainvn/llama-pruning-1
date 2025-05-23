from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Union
from torch import dtype
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from transformers import BitsAndBytesConfig

import torch

# Load the model and tokenizer.
def load_model(device: str = 'cuda', dtype: Optional[Union[str, dtype]] = torch.float32, cache_dir: Optional[str] = None, lora_path: str = None) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Loads the model and tokenizer.

    Args:
    - model_name: Name of the model to load.
    - dtype: Data type to use.
    - cache_dir: Directory to cache the model.

    Returns:
    - model: Model loaded.
    - tokenizer: Tokenizer loaded.
    """
    #Load the model and tokenizer.

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,  # Only this flag here
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="fp4",  # usually "nf4" or "fp4"
        bnb_4bit_compute_dtype=torch.float16  # or bfloat16
    )

    base_model = AutoModelForCausalLM.from_pretrained("unsloth/phi-4-bnb-4bit", torch_dtype=dtype, cache_dir=cache_dir, device_map=device, quantization_config=quant_config)
    tokenizer = AutoTokenizer.from_pretrained("unsloth/phi-4-bnb-4bit", use_fast=True)

    model = PeftModel.from_pretrained(base_model, lora_path)

    # Merge the model and tokenizer.
    model = model.merge_and_unload()

    return model, tokenizer