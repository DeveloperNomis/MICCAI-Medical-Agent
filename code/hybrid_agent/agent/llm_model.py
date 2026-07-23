import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LocalLLM:
    """
    Local causal language model wrapper used for routing, fallback parsing,
    and optional report generation.
    """

    def __init__(self, model_name=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_name = model_name or os.environ.get(
            "LOCAL_LLM_MODEL",
            "google/medgemma-4b-it",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=self.device,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )

        self.model.eval()

    def _tokenize(self, prompt: str):
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        )

        target_device = next(self.model.parameters()).device

        return {
            k: v.to(target_device)
            for k, v in inputs.items()
        }

    def generate(self, prompt: str, max_new_tokens: int = 100) -> str:
        """
        Generate text for fallback parsing, reports, or explanations.

        Only the newly generated text is returned.
        """
        inputs = self._tokenize(prompt)
        input_length = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_tokens = output_ids[0][input_length:]

        return self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

    def next_token_logits(self, prompt: str) -> torch.Tensor:
        """
        Return logits for the next token after the prompt.

        This is useful when the router scores task labels via next-token
        probabilities.

        Returns:
            Tensor of shape [vocab_size].
        """
        inputs = self._tokenize(prompt)

        with torch.inference_mode():
            outputs = self.model(**inputs)

        logits = outputs.logits[0, -1, :]

        return logits.detach().float().cpu()

    def score_next_token_options(self, prompt: str, options: list[str]) -> dict:
        """
        Score options that should ideally correspond to exactly one token.

        For multi-token labels, use score_text_options().
        """
        logits = self.next_token_logits(prompt)
        log_probs = torch.log_softmax(logits, dim=-1)

        scores = {}

        for option in options:
            token_ids = self.tokenizer.encode(
                option,
                add_special_tokens=False,
            )

            if len(token_ids) != 1:
                raise ValueError(
                    f"Option '{option}' is split into {len(token_ids)} tokens: {token_ids}. "
                    "Use score_text_options() for multi-token labels."
                )

            scores[option] = float(log_probs[token_ids[0]])

        return scores

    def score_text_options(self, prompt: str, options: list[str]) -> dict:
        """
        Score multi-token text options.

        This is usually more robust for task routing than next-token-only
        scoring. The score is the sum of log probabilities of the option tokens
        following the prompt.
        """
        scores = {}

        for option in options:
            full_text = prompt + option

            prompt_inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            )

            full_inputs = self.tokenizer(
                full_text,
                return_tensors="pt",
                add_special_tokens=False,
            )

            target_device = next(self.model.parameters()).device

            full_inputs = {
                k: v.to(target_device)
                for k, v in full_inputs.items()
            }

            prompt_len = prompt_inputs["input_ids"].shape[1]
            input_ids = full_inputs["input_ids"]

            with torch.inference_mode():
                outputs = self.model(**full_inputs)

            logits = outputs.logits

            # The probability of token t_i is taken from logits[i - 1].
            option_token_ids = input_ids[0, prompt_len:]
            option_logits = logits[0, prompt_len - 1:-1, :]

            log_probs = torch.log_softmax(option_logits.float(), dim=-1)

            token_scores = log_probs[
                torch.arange(option_token_ids.shape[0]),
                option_token_ids,
            ]

            scores[option] = float(token_scores.sum().detach().cpu())

        return scores

    def route(self, prompt: str, task_labels: list[str]) -> tuple[str, dict]:
        """
        Convenience method for the task router.

        Returns:
            The best label and all option scores.
        """
        scores = self.score_text_options(prompt, task_labels)
        best_label = max(scores, key=scores.get)

        return best_label, scores