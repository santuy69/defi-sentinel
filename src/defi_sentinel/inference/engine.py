"""LLM inference engine with multi-backend support."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Result from LLM inference."""
    content: str
    tokens: int
    latency_ms: float
    tokens_per_sec: float
    backend: str
    model: str


class LLMEngine:
    """Local LLM inference engine supporting CUDA, ROCm, and CPU backends.

    Automatically detects available hardware and selects the optimal backend.
    Supports GPTQ, AWQ, and GGUF quantization for memory-efficient inference.

    Usage:
        engine = LLMEngine(model="TheBloke/Llama-2-7B-Chat-GPTQ", backend="auto")
        result = engine.chat([{"role": "user", "content": "Analyze ETH/USDC pool"}])
    """

    BACKENDS = {
        "cuda": "torch.cuda.is_available",
        "rocm": "torch.cuda.is_available",  # ROCm uses torch.cuda API
        "cpu": "always_available",
    }

    def __init__(
        self,
        model: str = "TheBloke/Llama-2-7B-Chat-GPTQ",
        backend: str = "auto",
        quantization: str | None = None,
        device_map: str = "auto",
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
    ):
        self.model_name = model
        self.backend = self._detect_backend(backend)
        self.quantization = quantization
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._model = None
        self._tokenizer = None

        logger.info(
            "LLMEngine init — model=%s, backend=%s, quant=%s",
            model, self.backend, quantization or "default",
        )

    def _detect_backend(self, backend: str) -> str:
        """Auto-detect the best available backend."""
        if backend != "auto":
            return backend

        try:
            import torch
            if torch.cuda.is_available():
                # Check if ROCm by looking at torch version or hip
                if hasattr(torch.version, "hip") and torch.version.hip:
                    return "rocm"
                return "cuda"
        except ImportError:
            pass

        logger.warning("No GPU detected, falling back to CPU inference")
        return "cpu"

    def load(self) -> None:
        """Load model and tokenizer into memory."""
        if self._model is not None:
            return

        logger.info("Loading model %s on %s...", self.model_name, self.backend)
        start = time.time()

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            dtype = torch.float16 if self.backend != "cpu" else torch.float32

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )

            load_kwargs: dict[str, Any] = {
                "trust_remote_code": True,
                "torch_dtype": dtype,
            }

            if self.backend != "cpu":
                load_kwargs["device_map"] = self.device_map

            if self.quantization == "gptq-4bit":
                load_kwargs["quantization_config"] = {"bits": 4, "quant_method": "gptq"}

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **load_kwargs,
            )

            elapsed = time.time() - start
            logger.info("Model loaded in %.1fs on %s", elapsed, self.backend)

        except ImportError as e:
            logger.error("Missing dependencies: %s. Install with: pip install defi-sentinel[inference]", e)
            raise

    def chat(self, messages: list[dict[str, str]], **kwargs) -> dict[str, Any]:
        """Send a chat completion request to the local model.

        Args:
            messages: OpenAI-format message list.
            **kwargs: Override max_new_tokens, temperature.

        Returns:
            Dict with 'content', 'tokens', 'latency_ms', etc.
        """
        self.load()

        max_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        temp = kwargs.get("temperature", self.temperature)

        # Format prompt from messages
        prompt = self._format_prompt(messages)

        start = time.time()

        import torch
        inputs = self._tokenizer(prompt, return_tensors="pt")
        if self.backend != "cpu":
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temp,
                do_sample=temp > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        content = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        latency = (time.time() - start) * 1000
        token_count = len(new_tokens)
        tps = token_count / (latency / 1000) if latency > 0 else 0

        return {
            "content": content,
            "tokens": token_count,
            "latency_ms": round(latency, 1),
            "tokens_per_sec": round(tps, 1),
            "backend": self.backend,
            "model": self.model_name,
        }

    def _format_prompt(self, messages: list[dict[str, str]]) -> str:
        """Format messages into a single prompt string."""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"<<SYS>>\n{content}\n<</SYS>>")
            elif role == "user":
                parts.append(f"[INST] {content} [/INST]")
            elif role == "assistant":
                parts.append(content)
        return "\n".join(parts)

    def unload(self) -> None:
        """Free GPU memory."""
        if self._model is not None:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            logger.info("Model unloaded, GPU memory freed")
