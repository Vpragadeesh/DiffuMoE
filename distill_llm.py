"""
LLM Distillation with GGUF Teacher (Correct Tokenizer + Stable)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import logging
from pathlib import Path
from llama_cpp import Llama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# GGUF TEACHER
# ============================================================================

class GGUFTeacher:
    def __init__(self, model_path, n_ctx=512, n_gpu_layers=20, n_threads=6):
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            logits_all=True,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            verbose=False,
        )
        self.cache = {}

    def get_logits(self, input_ids):
        logits_batch = []

        for seq in input_ids:
            tokens = tuple(seq.tolist())

            if tokens in self.cache:
                logits = self.cache[tokens]
            else:
                try:
                    self.model.reset()
                    self.model.eval(tokens)

                    logits = torch.tensor(self.model._scores, dtype=torch.float32)

                    # Safety: ensure shape matches sequence
                    if logits.shape[0] != len(tokens):
                        logits = logits[:len(tokens)]

                    self.cache[tokens] = logits

                except Exception as e:
                    print("⚠️ GGUF error, skipping sequence:", e)
                    logits = torch.zeros(len(tokens), self.model.n_vocab())

            logits_batch.append(logits)

        return torch.stack(logits_batch)


# ============================================================================
# CONFIG
# ============================================================================

class DistillationConfig:
    def __init__(self):
        self.teacher_gguf_path = "/home/pragadeesh/model/mistral-7b-instruct-v0.2.Q2_K.gguf"

        self.student_hidden_dim = 512
        self.student_num_layers = 8
        self.student_num_heads = 8

        self.batch_size = 2
        self.gradient_accumulation_steps = 4
        self.learning_rate = 5e-4
        self.max_steps = 1000
        self.warmup_steps = 100

        self.temperature = 4.0
        self.max_seq_length = 128

        self.log_interval = 10


# ============================================================================
# DATASET
# ============================================================================

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=True
        )

        return {
            "input_ids": enc["input_ids"].squeeze()
        }


# ============================================================================
# STUDENT MODEL
# ============================================================================

class StudentModel(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, config.student_hidden_dim)
        self.pos_embedding = nn.Embedding(config.max_seq_length, config.student_hidden_dim)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.student_hidden_dim,
                nhead=config.student_num_heads,
                dim_feedforward=config.student_hidden_dim * 4,
                batch_first=True
            )
            for _ in range(config.student_num_layers)
        ])

        self.lm_head = nn.Linear(config.student_hidden_dim, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)

        pos = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        x = x + self.pos_embedding(pos)

        for block in self.blocks:
            x = block(x)

        return self.lm_head(x)


# ============================================================================
# LOSS
# ============================================================================

class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0):
        super().__init__()
        self.temperature = temperature
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits):
        s = F.log_softmax(student_logits / self.temperature, dim=-1)
        t = F.softmax(teacher_logits / self.temperature, dim=-1)
        return self.kl(s, t)


# ============================================================================
# TRAINER
# ============================================================================

class Trainer:
    def __init__(self, config, device):
        self.config = config
        self.device = device

        logger.info("Loading Mistral tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2"
        )

        # Fix padding
        self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Loading GGUF teacher...")
        self.teacher = GGUFTeacher(config.teacher_gguf_path)

        logger.info("Creating student...")
        self.student = StudentModel(
            config,
            self.tokenizer.vocab_size
        ).to(device)

        self.optimizer = AdamW(self.student.parameters(), lr=config.learning_rate)

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            config.warmup_steps,
            config.max_steps
        )

        self.criterion = DistillationLoss(config.temperature)

        self.step = 0

    def train_step(self, batch):
        input_ids = batch["input_ids"].to(self.device)

        student_logits = self.student(input_ids)

        with torch.no_grad():
            teacher_logits = self.teacher.get_logits(input_ids).to(self.device)

        # Match sequence length (safety)
        min_len = min(student_logits.shape[1], teacher_logits.shape[1])
        student_logits = student_logits[:, :min_len, :]
        teacher_logits = teacher_logits[:, :min_len, :]

        loss = self.criterion(student_logits, teacher_logits)

        loss.backward()

        if self.step % self.config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        self.step += 1
        return loss.item()

    def train(self, dataloader):
        self.student.train()

        while self.step < self.config.max_steps:
            for batch in dataloader:
                loss = self.train_step(batch)

                if self.step % self.config.log_interval == 0:
                    logger.info(f"Step {self.step} | Loss: {loss:.4f}")

                if self.step >= self.config.max_steps:
                    break

        Path("checkpoints").mkdir(exist_ok=True)
        torch.save(self.student.state_dict(), "checkpoints/student.pt")

        logger.info("Training complete!")


# ============================================================================
# MAIN
# ============================================================================

def main():
    config = DistillationConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(config, device)

    texts = ["AI is transforming the world." * 10 for _ in range(200)]

    dataset = TextDataset(texts, trainer.tokenizer, config.max_seq_length)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    trainer.train(dataloader)


if __name__ == "__main__":
    main()
