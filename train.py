#!/usr/bin/env python3
from qwen_distill import QwenDistillationConfig, QwenDistillationTrainer, TextDataset, load_training_texts
from torch.utils.data import DataLoader
import torch

# Load config
config = QwenDistillationConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize trainer
trainer = QwenDistillationTrainer(config, device)

# Load data
texts = load_training_texts(config.data_file)

print(f"Loaded {len(texts)} cleaned text samples from {config.data_file}")

# Create dataset & dataloader
dataset = TextDataset(texts, trainer.tokenizer, max_length=config.max_seq_length)
dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)

# Train
trainer.train(dataloader)

print("✓ Training complete!")
print(f"Student saved to: checkpoints/student_final.pt")
