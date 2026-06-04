"""LLM prompting for antisemitism detection (Models D1, D2, D3)."""
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class LLMPrompter:
    """
    Zero-shot, few-shot, and Guided-CoT prompting for antisemitism detection.

    Supports 4-bit quantization for running 8B models on a single GPU.
    """

    VALID_LABELS = {"antisemitic", "not antisemitic"}
    FINAL_LABEL_PATTERNS = [
        re.compile(r"final classification:\s*(antisemitic|not antisemitic)\s*$", re.IGNORECASE),
        re.compile(r"classification:\s*(antisemitic|not antisemitic)\s*$", re.IGNORECASE),
        re.compile(r"^\s*(antisemitic|not antisemitic)\s*$", re.IGNORECASE),
    ]

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        prompt_template: str = "zero_shot",
        quantization: str = "4bit",
        max_new_tokens: int = 50,
        temperature: float = 0.0,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device

        # Load prompt template
        prompt_dir = Path(__file__).parent / "prompts"
        self.template = (prompt_dir / f"{prompt_template}.txt").read_text()

        # Load model with quantization
        logger.info(f"Loading {model_name} with {quantization} quantization...")

        if quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Model loaded successfully.")

    def _format_prompt(self, text: str) -> str:
        """Insert tweet text into prompt template."""
        return self.template.replace("{text}", text)

    def _parse_response(self, response: str) -> str:
        """
        Extract classification label from model response.

        Handles various response formats:
        - Preferred: final line `Final classification: <label>`
        - Fallback: final line `Classification: <label>`
        - Last resort: a line containing only the bare label

        We intentionally avoid substring matching over the whole response,
        because CoT outputs can mention both labels while reasoning.
        """
        lines = [line.strip() for line in response.splitlines() if line.strip()]

        for line in reversed(lines):
            for pattern in self.FINAL_LABEL_PATTERNS:
                match = pattern.match(line)
                if match:
                    label = match.group(1).lower()
                    if label in self.VALID_LABELS:
                        return label

        return "invalid"

    @torch.no_grad()
    def classify_batch(self, texts: list[str]) -> list[dict]:
        """
        Classify a batch of texts.

        Returns list of dicts with keys: 'prediction', 'raw_response', 'is_valid'
        """
        results = []

        for text in tqdm(texts, desc="LLM inference"):
            prompt = self._format_prompt(text)

            # Format as chat for instruct models
            messages = [{"role": "user", "content": prompt}]

            input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            # Decode only the new tokens
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True,
            )

            label = self._parse_response(response)

            results.append({
                "prediction": label,
                "raw_response": response,
                "is_valid": label != "invalid",
            })

        return results
