#!/usr/bin/env python3
"""
Run a distilled student checkpoint for text generation.
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from qwen_distill import QwenDistillationConfig, QwenStudentModel


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class StudentRunner:
    """Load a trained student checkpoint and generate text."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str | None = None,
        tokenizer_path: str | None = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        config_data = checkpoint["config"]

        config = QwenDistillationConfig()
        for key, value in config_data.items():
            setattr(config, key, value)
        self.config = config

        self.model = QwenStudentModel(self.config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        tokenizer_source = self._resolve_tokenizer_source(tokenizer_path)
        logger.info("Loading tokenizer from %s", tokenizer_source)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=True,
            local_files_only=Path(tokenizer_source).exists(),
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(
            "Loaded student checkpoint from %s on %s",
            self.checkpoint_path,
            self.device,
        )

    def _resolve_tokenizer_source(self, tokenizer_path: str | None) -> str:
        if tokenizer_path:
            return tokenizer_path

        local_teacher = Path("models/teacher")
        if local_teacher.exists():
            return str(local_teacher)

        return self.config.teacher_model_name

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
    ) -> str:
        if not prompt.strip():
            raise ValueError("Prompt must not be empty.")

        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = encoded["input_ids"].to(self.device)

        with torch.inference_mode():
            for _ in range(max_new_tokens):
                window = input_ids[:, -self.config.max_seq_length :]
                attention_mask = torch.ones_like(window, device=self.device)

                outputs = self.model(window, attention_mask=attention_mask)
                next_token_logits = outputs["logits"][:, -1, :]
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits,
                    input_ids,
                    repetition_penalty,
                )
                next_token = self._sample_token(
                    next_token_logits,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
                input_ids = torch.cat([input_ids, next_token], dim=-1)

                if self.tokenizer.eos_token_id is not None and next_token.item() == self.tokenizer.eos_token_id:
                    break

        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        repetition_penalty: float,
    ) -> torch.Tensor:
        if repetition_penalty <= 1.0:
            return logits

        adjusted = logits.clone()
        for token_id in torch.unique(input_ids):
            token_index = token_id.item()
            token_score = adjusted[:, token_index]
            adjusted[:, token_index] = torch.where(
                token_score < 0,
                token_score * repetition_penalty,
                token_score / repetition_penalty,
            )
        return adjusted

    @staticmethod
    def _sample_token(
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> torch.Tensor:
        if temperature <= 0:
            return torch.argmax(logits, dim=-1, keepdim=True)

        scaled_logits = logits / temperature

        if top_k > 0:
            top_k = min(top_k, scaled_logits.shape[-1])
            values, _ = torch.topk(scaled_logits, top_k)
            cutoff = values[:, -1].unsqueeze(-1)
            scaled_logits = torch.where(
                scaled_logits < cutoff,
                torch.full_like(scaled_logits, float("-inf")),
                scaled_logits,
            )

        if 0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            sorted_mask = cumulative_probs > top_p
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False

            removal_mask = torch.zeros_like(sorted_mask, dtype=torch.bool)
            removal_mask.scatter_(dim=-1, index=sorted_indices, src=sorted_mask)
            scaled_logits = scaled_logits.masked_fill(removal_mask, float("-inf"))

        probs = F.softmax(scaled_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a trained student checkpoint.")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/student_final.pt",
        help="Path to the student checkpoint.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run on. Defaults to cuda if available, otherwise cpu.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Optional tokenizer path. Defaults to models/teacher if present.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt to generate from.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature. Use 0 for greedy decoding.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling threshold.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling cutoff. Use 0 to disable.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Penalty for already generated tokens. Use 1.0 to disable.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start an interactive prompt loop.",
    )
    return parser


def interactive_loop(runner: StudentRunner, args: argparse.Namespace) -> None:
    print("Interactive mode. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            prompt = input("\nPrompt> ").strip()
        except EOFError:
            print()
            break

        if prompt.lower() in {"exit", "quit"}:
            break
        if not prompt:
            continue

        output = runner.generate(
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )
        print(f"\n{output}")


def main() -> None:
    args = build_parser().parse_args()
    runner = StudentRunner(
        checkpoint_path=args.checkpoint,
        device=args.device,
        tokenizer_path=args.tokenizer_path,
    )

    if args.interactive:
        interactive_loop(runner, args)
        return

    if not args.prompt:
        raise SystemExit("Provide --prompt for one-shot generation or use --interactive.")

    output = runner.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )
    print(output)


if __name__ == "__main__":
    main()
