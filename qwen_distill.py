"""
LLM Distillation: Qwen3.5-0.8B → Student (100-150M)
Adapted for RTX 2050, Arch Linux, integrated with DiffuMoE
"""

import argparse
import json
import logging
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIG
# ============================================================================

class QwenDistillationConfig:
    """Configuration for Qwen-0.8B → Student distillation"""
    def __init__(self):
        # Teacher: Qwen3.5-0.8B
        self.teacher_model_name = "Qwen/Qwen2.5-0.5B"  # Base Qwen (closest to your 0.8B)
        # Alternative: "Qwen/Qwen1.5-0.5B" if above unavailable
        
        # Student: 100-150M params (4-5 layers × 256 hidden)
        self.student_hidden_dim = 256      # Smaller than teacher's 1024
        self.student_num_layers = 5        # Qwen has 24 layers, student: 5
        self.student_num_heads = 4         # 256 / 4 = 64 per head
        self.student_head_dim = 64
        self.vocab_size = 151936           # Qwen tokenizer vocab
        
        # Architecture
        self.max_seq_length = 256          # Smaller sequences for RTX 2050
        self.hidden_act = "silu"           # Use Qwen's activation (or gelu)
        
        # Distillation hyperparameters
        self.temperature = 3.0             # Smaller teacher → lower temperature
        self.alpha = 0.8                   # KD loss weight (response-based)
        self.beta = 0.2                    # Feature loss weight (hidden state matching)
        self.feature_loss_type = "cosine"  # "mse" or "cosine"
        self.kd_chunk_tokens = 16          # Chunk softmax/KL over sequence to reduce VRAM
        self.lm_loss_weight = 1.0          # Next-token LM loss for better English generation
        
        # Training
        self.batch_size = 1                # Safer default for 4GB GPUs
        self.gradient_accumulation_steps = 8  # Keep effective batch close to previous default (1 × 8 = 8)
        self.learning_rate = 8e-4
        self.weight_decay = 0.01
        self.warmup_steps = 100
        self.max_steps = 2000              # Smaller teacher = fewer steps needed
        self.save_steps = 200
        self.eval_steps = 200
        
        # Memory optimization
        self.use_gradient_checkpointing = True
        self.use_flash_attention = True    # If available
        self.mixed_precision = "fp16"      # fp16 or bf16
        self.data_file = "data/train.txt"
        
        # Logging
        self.log_interval = 20
        self.experiment_name = "qwen_0.8b_distillation"


# ============================================================================
# DATASET
# ============================================================================

class TextDataset(Dataset):
    """Simple text dataset for distillation"""
    def __init__(self, texts: list, tokenizer, max_length: int = 256):
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
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze() if "attention_mask" in enc else torch.ones(self.max_length),
        }


HEADING_RE = re.compile(r"^\s*=+.*=+\s*$")


def clean_training_text(text: str) -> str:
    """Normalize common WikiText artifacts into more natural English text."""
    text = text.replace(" @-@ ", "-")
    text = text.replace(" @,@ ", ",")
    text = text.replace(" @.@ ", ".")
    text = text.replace(" ; ", "; ")
    text = text.replace(" : ", ": ")
    text = text.replace(" 's", "'s")
    text = text.replace(" 't", "'t")
    text = text.replace(" 're", "'re")
    text = text.replace(" 've", "'ve")
    text = text.replace(" 'm", "'m")
    text = text.replace(" 'll", "'ll")
    text = text.replace(" 'd", "'d")
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    text = re.sub(r"\s+([\)\]\}])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def load_training_texts(data_file: str, min_chars: int = 40, max_samples: int | None = None) -> list[str]:
    """Load paragraph-level text samples from a corpus file."""
    path = Path(data_file)
    if not path.exists():
        raise FileNotFoundError(f"Training data file not found: {path}")

    texts = []
    paragraph_lines = []

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        if not paragraph_lines:
            return
        text = clean_training_text(" ".join(paragraph_lines))
        if len(text) >= min_chars:
            texts.append(text)
        paragraph_lines = []

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                flush_paragraph()
                continue
            if HEADING_RE.fullmatch(line):
                flush_paragraph()
                continue
            paragraph_lines.append(line)

    flush_paragraph()

    if max_samples is not None:
        texts = texts[:max_samples]
    if not texts:
        raise RuntimeError(f"No usable training samples found in {path}")

    return texts


# ============================================================================
# STUDENT MODEL (Lightweight)
# ============================================================================

class QwenStudentModel(nn.Module):
    """
    Lightweight Qwen-style student model (100-150M params)
    - 5 decoder layers
    - 256 hidden dim
    - 4 heads
    - Efficient rotary embeddings (RoPE)
    """
    
    def __init__(self, config: QwenDistillationConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.embedding = nn.Embedding(config.vocab_size, config.student_hidden_dim)
        
        # Rotary position embeddings (RoPE) - Qwen style
        # Simplified: use absolute positional embeddings instead
        self.pos_embedding = nn.Embedding(config.max_seq_length, config.student_hidden_dim)
        
        # Decoder blocks with layer norm
        self.layers = nn.ModuleList([
            QwenDecoderLayer(config) for _ in range(config.student_num_layers)
        ])
        
        self.final_ln = nn.LayerNorm(config.student_hidden_dim)
        self.lm_head = nn.Linear(config.student_hidden_dim, config.vocab_size, bias=False)
        
        logger.info(f"Student: {config.student_num_layers} layers, {config.student_hidden_dim} hidden, "
                   f"{self._count_params() / 1e6:.1f}M params")
    
    def _count_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        
        # Add positional embeddings
        seq_len = input_ids.shape[1]
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = x + self.pos_embedding(pos_ids)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        
        # Pass through decoder layers, collecting hidden states
        hidden_states = [x]
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask, causal_mask=causal_mask)
            hidden_states.append(x)
        
        # Final layer norm and logits
        x = self.final_ln(x)
        logits = self.lm_head(x)
        
        return {
            'logits': logits,
            'hidden_states': hidden_states,
        }


class QwenDecoderLayer(nn.Module):
    """Single Qwen decoder layer"""
    def __init__(self, config: QwenDistillationConfig):
        super().__init__()
        self.hidden_size = config.student_hidden_dim
        self.num_heads = config.student_num_heads
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.student_hidden_dim,
            num_heads=config.student_num_heads,
            dropout=0.1,
            batch_first=True,
        )
        
        # MLP (feed-forward)
        self.mlp = nn.Sequential(
            nn.Linear(config.student_hidden_dim, config.student_hidden_dim * 4),
            nn.GELU(),
            nn.Linear(config.student_hidden_dim * 4, config.student_hidden_dim),
            nn.Dropout(0.1),
        )
        
        # Layer norms
        self.ln1 = nn.LayerNorm(config.student_hidden_dim)
        self.ln2 = nn.LayerNorm(config.student_hidden_dim)
    
    def forward(self, x, attention_mask=None, causal_mask=None):
        # Self-attention with residual
        attn_out, _ = self.self_attn(
            self.ln1(x), self.ln1(x), self.ln1(x),
            attn_mask=causal_mask,
            key_padding_mask=~attention_mask.bool() if attention_mask is not None else None,
            need_weights=False,
        )
        x = x + attn_out
        
        # MLP with residual
        mlp_out = self.mlp(self.ln2(x))
        x = x + mlp_out
        
        return x


# ============================================================================
# DISTILLATION LOSS
# ============================================================================

class QwenDistillationLoss(nn.Module):
    """Response-based + Feature-based KD loss"""
    
    def __init__(self, config: QwenDistillationConfig):
        super().__init__()
        self.config = config
        self.temperature = config.temperature
        self.alpha = config.alpha
        self.beta = config.beta
    
    def forward(self, student_logits, teacher_logits, student_hidden, teacher_hidden, attention_mask=None, labels=None):
        """
        Compute combined KD loss
        
        Args:
            student_logits: (B, T, V) student output logits
            teacher_logits: (B, T, V) teacher output logits
            student_hidden: list of (B, T, D_s) hidden states
            teacher_hidden: list of (B, T, D_t) hidden states
            attention_mask: (B, T) attention mask
        """
        
        # Response-based KD (soft targets), computed in chunks to reduce peak VRAM.
        kd_loss = self._kd_loss_chunked(student_logits, teacher_logits, attention_mask)
        
        # Feature-based distillation (match hidden layers)
        feature_loss = 0.0
        if self.beta > 0 and len(student_hidden) > 0:
            feature_loss = self._feature_loss(student_hidden, teacher_hidden, attention_mask)

        lm_loss = 0.0
        if self.config.lm_loss_weight > 0 and labels is not None:
            lm_loss = self._lm_loss_chunked(student_logits, labels, attention_mask)
        
        # Total loss
        total_loss = (
            self.alpha * kd_loss
            + self.beta * feature_loss
            + self.config.lm_loss_weight * lm_loss
        )
        
        return {
            'total': total_loss,
            'kd': kd_loss.item(),
            'feature': feature_loss.item() if isinstance(feature_loss, torch.Tensor) else feature_loss,
            'lm': lm_loss.item() if isinstance(lm_loss, torch.Tensor) else lm_loss,
        }

    def _kd_loss_chunked(self, student_logits, teacher_logits, attention_mask=None):
        """
        Compute token-level KL in sequence chunks to avoid materializing full-vocab
        softmax tensors for the entire sequence at once.
        """
        _, seq_len, _ = student_logits.shape
        chunk_tokens = max(1, int(getattr(self.config, "kd_chunk_tokens", 16)))

        total_kl = student_logits.new_zeros(())
        total_tokens = student_logits.new_zeros(())

        for start in range(0, seq_len, chunk_tokens):
            end = min(seq_len, start + chunk_tokens)

            s_chunk = student_logits[:, start:end, :] / self.temperature
            t_chunk = teacher_logits[:, start:end, :] / self.temperature

            log_probs_student = F.log_softmax(s_chunk, dim=-1)
            probs_teacher = F.softmax(t_chunk, dim=-1)
            token_kl = F.kl_div(log_probs_student, probs_teacher, reduction="none").sum(dim=-1)

            if attention_mask is not None:
                mask = attention_mask[:, start:end].to(token_kl.dtype)
                total_kl = total_kl + (token_kl * mask).sum()
                total_tokens = total_tokens + mask.sum()
            else:
                total_kl = total_kl + token_kl.sum()
                total_tokens = total_tokens + token_kl.new_tensor(float(token_kl.numel()))

        return total_kl / total_tokens.clamp_min(1.0)

    def _lm_loss_chunked(self, student_logits, labels, attention_mask=None):
        """Compute next-token CE in chunks for stability and lower VRAM."""
        if student_logits.shape[1] < 2:
            return student_logits.new_zeros(())

        shift_logits = student_logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        shift_mask = attention_mask[:, 1:] if attention_mask is not None else None
        chunk_tokens = max(1, int(getattr(self.config, "kd_chunk_tokens", 16)))

        total_loss = student_logits.new_zeros(())
        total_tokens = student_logits.new_zeros(())

        for start in range(0, shift_logits.shape[1], chunk_tokens):
            end = min(shift_logits.shape[1], start + chunk_tokens)
            chunk_logits = shift_logits[:, start:end, :].reshape(-1, shift_logits.shape[-1]).float()
            chunk_labels = shift_labels[:, start:end].reshape(-1)

            if shift_mask is not None:
                chunk_mask = shift_mask[:, start:end].reshape(-1).bool()
            else:
                chunk_mask = torch.ones_like(chunk_labels, dtype=torch.bool)

            if chunk_mask.any():
                total_loss = total_loss + F.cross_entropy(
                    chunk_logits[chunk_mask],
                    chunk_labels[chunk_mask],
                    reduction="sum",
                )
                total_tokens = total_tokens + chunk_mask.sum()

        return total_loss / total_tokens.clamp_min(1)

    @staticmethod
    def _pool_last_dim(hidden: torch.Tensor, target_dim: int) -> torch.Tensor:
        """Resize hidden dimension (last axis) with parameter-free average pooling."""
        bsz, seq_len, hidden_dim = hidden.shape
        if hidden_dim == target_dim:
            return hidden

        pooled = F.adaptive_avg_pool1d(
            hidden.reshape(bsz * seq_len, 1, hidden_dim),
            target_dim,
        )
        return pooled.reshape(bsz, seq_len, target_dim)
    
    def _feature_loss(self, student_hidden, teacher_hidden, attention_mask):
        """Match intermediate layer representations"""
        loss = 0.0
        num_layers = min(len(student_hidden), len(teacher_hidden))
        
        for i in range(num_layers):
            s_hidden = student_hidden[i]  # (B, T, D_s)
            t_hidden = teacher_hidden[i]  # (B, T, D_t)

            # Align hidden dimensions before feature matching.
            if s_hidden.shape[-1] != t_hidden.shape[-1]:
                target_dim = min(s_hidden.shape[-1], t_hidden.shape[-1])
                s_hidden = self._pool_last_dim(s_hidden, target_dim)
                t_hidden = self._pool_last_dim(t_hidden, target_dim)
            
            # Cosine similarity loss or MSE
            if self.config.feature_loss_type == "cosine":
                s_norm = F.normalize(s_hidden, p=2, dim=-1)
                t_norm = F.normalize(t_hidden, p=2, dim=-1)
                loss += (1 - F.cosine_similarity(s_norm, t_norm, dim=-1)).mean()
            else:
                loss += F.mse_loss(s_hidden, t_hidden)
        
        return loss / num_layers if num_layers > 0 else torch.tensor(0.0, device=student_hidden[0].device)


# ============================================================================
# TRAINER
# ============================================================================

class QwenDistillationTrainer:
    """Main training loop for Qwen distillation"""
    
    def __init__(self, config: QwenDistillationConfig, device: torch.device):
        self.config = config
        self.device = device
        
        # Load tokenizer
        logger.info(f"Loading Qwen tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.teacher_model_name,
            trust_remote_code=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load teacher
        logger.info(f"Loading teacher: {config.teacher_model_name}")
        self.teacher = AutoModelForCausalLM.from_pretrained(
            config.teacher_model_name,
            dtype=torch.float16 if config.mixed_precision == "fp16" else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        self.teacher.config.use_cache = False
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # Create student
        logger.info(f"Creating student model...")
        self.student = QwenStudentModel(config).to(device)
        
        # Optimizer & scheduler
        self.optimizer = AdamW(
            self.student.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.max_steps,
        )
        
        # Loss
        self.criterion = QwenDistillationLoss(config)
        
        # Metrics
        self.history = {
            'step': [],
            'loss': [],
            'kd_loss': [],
            'feature_loss': [],
            'lm_loss': [],
            'learning_rate': [],
        }
        self.global_step = 0
        self.use_amp = self.device.type == "cuda" and self.config.mixed_precision in {"fp16", "bf16"}
        self.amp_dtype = torch.float16 if self.config.mixed_precision == "fp16" else torch.bfloat16
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp and self.amp_dtype == torch.float16)
        self.optimizer.zero_grad(set_to_none=True)
        
        logger.info(f"✓ Setup complete. Device: {device}")
    
    def train_step(self, batch):
        """Single training step"""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        # Student forward
        with torch.autocast(
            device_type="cuda",
            dtype=self.amp_dtype,
            enabled=self.use_amp,
        ):
            student_output = self.student(input_ids, attention_mask)
            student_logits = student_output['logits']
            student_hidden = student_output['hidden_states']
        
        # Teacher forward (no grad)
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda",
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                teacher_output = self.teacher(
                    input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )
                teacher_logits = teacher_output.logits
                teacher_hidden = teacher_output.hidden_states
        
        # Match sequence length
        min_len = min(student_logits.shape[1], teacher_logits.shape[1])
        student_logits = student_logits[:, :min_len, :]
        teacher_logits = teacher_logits[:, :min_len, :]
        input_ids = input_ids[:, :min_len]
        attention_mask = attention_mask[:, :min_len]
        
        # Compute loss
        loss_dict = self.criterion(
            student_logits,
            teacher_logits,
            [h[:, :min_len, :] for h in student_hidden],
            [h[:, :min_len, :] for h in teacher_hidden],
            attention_mask,
            labels=input_ids,
        )
        
        loss = loss_dict['total'] / self.config.gradient_accumulation_steps
        
        # Backward
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Optimizer step (with accumulation)
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            if self.scaler.is_enabled():
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
        
        self.global_step += 1
        
        return loss_dict
    
    def train(self, dataloader):
        """Main training loop"""
        self.student.train()
        dataloader_iter = iter(dataloader)
        
        logger.info(f"Starting training for {self.config.max_steps} steps...")
        
        try:
            while self.global_step < self.config.max_steps:
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    dataloader_iter = iter(dataloader)
                    batch = next(dataloader_iter)
                
                loss_dict = self.train_step(batch)
                
                # Log metrics
                if self.global_step % self.config.log_interval == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    total_loss_value = loss_dict['total'].item() if isinstance(loss_dict['total'], torch.Tensor) else float(loss_dict['total'])
                    logger.info(
                        f"Step {self.global_step}/{self.config.max_steps} | "
                        f"Loss: {total_loss_value:.4f} | "
                        f"KD: {loss_dict['kd']:.4f} | "
                        f"Feature: {loss_dict['feature']:.4f} | "
                        f"LM: {loss_dict['lm']:.4f} | "
                        f"LR: {lr:.2e}"
                    )
                    
                    self.history['step'].append(self.global_step)
                    self.history['loss'].append(total_loss_value)
                    self.history['kd_loss'].append(loss_dict['kd'])
                    self.history['feature_loss'].append(loss_dict['feature'])
                    self.history['lm_loss'].append(loss_dict['lm'])
                    self.history['learning_rate'].append(lr)
                
                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self._save_checkpoint()
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        # Final save
        self._save_checkpoint(final=True)
    
    def _save_checkpoint(self, final=False):
        """Save checkpoint"""
        ckpt_dir = Path("checkpoints")
        ckpt_dir.mkdir(exist_ok=True)
        
        if final:
            path = ckpt_dir / "student_final.pt"
        else:
            path = ckpt_dir / f"student_step_{self.global_step}.pt"
        
        torch.save({
            'model_state_dict': self.student.state_dict(),
            'config': self.config.__dict__,
            'global_step': self.global_step,
            'history': self.history,
        }, path)
        
        logger.info(f"✓ Checkpoint saved: {path}")
        
        # Also save metrics
        metrics_path = path.parent / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.history, f, indent=2)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train the distilled student model.")
    parser.add_argument("--data-file", default=None, help="Path to the training text file.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on number of training samples.")
    args = parser.parse_args()

    config = QwenDistillationConfig()
    if args.data_file:
        config.data_file = args.data_file
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Device: {device}")
    logger.info(f"Config: {json.dumps(config.__dict__, indent=2, default=str)}")
    
    # Initialize trainer
    trainer = QwenDistillationTrainer(config, device)
    
    logger.info("Preparing dataset...")
    texts = load_training_texts(config.data_file, max_samples=args.max_samples)
    
    dataset = TextDataset(texts, trainer.tokenizer, max_length=config.max_seq_length)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    logger.info(f"Dataset size: {len(dataset)} from {config.data_file}")
    
    # Train
    trainer.train(dataloader)
    
    logger.info("✓ Training complete!")


if __name__ == "__main__":
    main()
