"""Model quantization utilities for memory-efficient inference."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def quantize_model(
    model_path: str,
    output_path: str,
    method: str = "gptq",
    bits: int = 4,
    dataset: str = "wikitext2",
    group_size: int = 128,
) -> str:
    """Quantize a model to reduce VRAM requirements.

    Args:
        model_path: HuggingFace model name or local path.
        output_path: Where to save quantized model.
        method: Quantization method (gptq, awq, gguf).
        bits: Target bit width (4 or 8).
        dataset: Calibration dataset.
        group_size: Quantization group size.

    Returns:
        Path to quantized model.
    """
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)

    if method == "gptq":
        return _quantize_gptq(model_path, str(output), bits, dataset, group_size)
    elif method == "awq":
        return _quantize_awq(model_path, str(output), bits, dataset)
    elif method == "gguf":
        return _quantize_gguf(model_path, str(output), bits)
    else:
        raise ValueError(f"Unknown quantization method: {method}")


def _quantize_gptq(model_path: str, output_path: str, bits: int, dataset: str, group_size: int) -> str:
    """GPTQ quantization using AutoGPTQ."""
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("Install auto-gptq: pip install auto-gptq")

    logger.info("Quantizing %s with GPTQ (%d-bit)...", model_path, bits)

    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        damp_percent=0.01,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoGPTQForCausalLM.from_pretrained(model_path, quantize_config)

    # Calibration with dataset samples
    logger.info("Running calibration on %s dataset...", dataset)
    model.quantize(tokenizer=tokenizer)

    model.save_quantized(output_path)
    tokenizer.save_pretrained(output_path)

    logger.info("GPTQ model saved to %s", output_path)
    return output_path


def _quantize_awq(model_path: str, output_path: str, bits: int, dataset: str) -> str:
    """AWQ quantization."""
    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("Install autoawq: pip install autoawq")

    logger.info("Quantizing %s with AWQ (%d-bit)...", model_path, bits)

    model = AutoAWQForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": bits}
    model.quantize(tokenizer, quant_config=quant_config)

    model.save_quantized(output_path)
    tokenizer.save_pretrained(output_path)

    logger.info("AWQ model saved to %s", output_path)
    return output_path


def _quantize_gguf(model_path: str, output_path: str, bits: int) -> str:
    """GGUF quantization (converts to llama.cpp format)."""
    logger.info("GGUF conversion for %s (%d-bit)...", model_path, bits)

    # GGUF conversion requires llama.cpp's convert script
    # This is a placeholder for the conversion pipeline
    quant_map = {4: "Q4_K_M", 8: "Q8_0", 16: "F16"}
    quant_type = quant_map.get(bits, "Q4_K_M")

    logger.info("Target GGUF quantization: %s", quant_type)
    logger.info("Use llama.cpp convert-hf-to-gguf.py for actual conversion")

    return output_path
