#!/usr/bin/env python3
"""
QUICK START: Qwen3.5-0.8B → Student (100-150M)
For RTX 2050 (4GB VRAM) on Arch Linux
"""

import subprocess
import sys
from pathlib import Path
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# STEP 0: Install Dependencies
# ============================================================================

def install_dependencies():
    """Install required packages with uv"""
    logger.info("Installing dependencies with uv...")
    
    packages = [
        "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        "transformers>=4.40.0",
        "accelerate",
        "datasets",
        "bitsandbytes",  # For quantization
        "peft",  # For LoRA
    ]
    
    for pkg in packages:
        logger.info(f"Installing: {pkg}")
        subprocess.run([sys.executable, "-m", "pip", "install", pkg], check=False)
    
    logger.info("✓ Dependencies installed")


# ============================================================================
# STEP 1: GGUF to HuggingFace Conversion
# ============================================================================

def convert_gguf_to_hf(gguf_path: str, output_dir: str = "models/qwen_teacher"):
    """
    Convert GGUF to HuggingFace format
    Note: This requires the model architecture config
    
    For Qwen3.5-0.8B, we can also just download from HuggingFace instead
    """
    logger.info(f"Converting GGUF: {gguf_path}")
    
    # Option 1: Use ollama/llama.cpp to load and export
    try:
        from llama_cpp import Llama
        logger.info("Loading GGUF with llama.cpp...")
        
        llm = Llama(model_path=gguf_path, n_gpu_layers=-1)
        # Note: llama.cpp doesn't easily export to HuggingFace format
        logger.warning("GGUF loading for inference only. For training, use HuggingFace model instead.")
        return llm
    
    except ImportError:
        logger.error("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
        logger.info("Alternative: Download Qwen from HuggingFace")
        return None


# ============================================================================
# STEP 2: Download Teacher Model
# ============================================================================

def download_qwen_teacher(output_dir: str = "models/teacher"):
    """Download Qwen teacher from HuggingFace"""
    logger.info("Downloading Qwen teacher model...")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_name = "Qwen/Qwen2.5-0.5B"  # Use 0.5B as proxy for 0.8B
    # Alternative options:
    # - "Qwen/Qwen1.5-0.5B"
    # - "Qwen/Qwen2-0.5B"
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
    )
    model.save_pretrained(output_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"✓ Model saved to {output_dir}")
    return output_dir


# ============================================================================
# STEP 3: Prepare Training Data
# ============================================================================

def prepare_dataset(dataset_name: str = "wikitext", split: str = "train", output_file: str = "data/train.txt"):
    """Download and prepare training data"""
    logger.info(f"Preparing dataset: {dataset_name}")
    
    from datasets import DownloadConfig, load_dataset
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading {dataset_name}...")
    if dataset_name == "wikitext":
        # Prefer canonical repo/config names and retry transient network failures.
        wikitext_candidates = [
            ("Salesforce/wikitext", "wikitext-2-raw-v1"),
            ("Salesforce/wikitext", "wikitext-2-v1"),
            ("wikitext", "wikitext-2-raw-v1"),
            ("wikitext", "wikitext-2"),
        ]
        max_attempts = 4
        backoff_seconds = 2
        download_config = DownloadConfig(max_retries=8)

        texts = None
        last_error = None
        for dataset_id, config_name in wikitext_candidates:
            for attempt in range(1, max_attempts + 1):
                try:
                    logger.info(
                        "Loading %s (%s), split=%s [attempt %s/%s]",
                        dataset_id,
                        config_name,
                        split,
                        attempt,
                        max_attempts,
                    )
                    dataset_split = load_dataset(
                        dataset_id,
                        config_name,
                        split=split,
                        download_config=download_config,
                    )
                    texts = dataset_split["text"]
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt < max_attempts:
                        sleep_s = backoff_seconds * attempt
                        logger.warning(
                            "Dataset load failed for %s (%s): %s. Retrying in %ss...",
                            dataset_id,
                            config_name,
                            exc,
                            sleep_s,
                        )
                        time.sleep(sleep_s)
            if texts is not None:
                break

        if texts is None:
            raise RuntimeError(
                "Failed to load WikiText after retries/fallbacks. "
                "Please check internet connectivity and Hugging Face availability."
            ) from last_error
    elif dataset_name == "pile":
        dataset = load_dataset("the_pile", split=f"{split}[:5000]")  # Subset
        texts = dataset["text"]
    else:
        logger.error(f"Unknown dataset: {dataset_name}")
        return None
    
    # Save to text file
    logger.info(f"Writing to {output_file}...")
    with open(output_file, 'w') as f:
        for text in texts:
            if text.strip():
                f.write(text + "\n")
    
    logger.info(f"✓ Dataset saved: {output_file}")
    return output_file


# ============================================================================
# STEP 4: Configuration
# ============================================================================

def create_config_template():
    """Create config.py template"""
    config_content = '''
# config.py - Training configuration
from qwen_distill import QwenDistillationConfig

class MyConfig(QwenDistillationConfig):
    def __init__(self):
        super().__init__()
        
        # Paths
        self.data_file = "data/train.txt"
        self.teacher_model_name = "Qwen/Qwen2.5-0.5B"
        
        # Student size (adjust based on your needs)
        # Small: 3 layers, 128 hidden = ~30M params
        # Medium: 5 layers, 256 hidden = ~100M params
        # Large: 8 layers, 384 hidden = ~250M params
        
        self.student_num_layers = 5
        self.student_hidden_dim = 256
        self.student_num_heads = 4
        
        # Training
        self.batch_size = 2
        self.gradient_accumulation_steps = 4
        self.max_steps = 2000
        self.learning_rate = 8e-4
        
        # Distillation
        self.temperature = 3.0
        self.alpha = 0.8  # 80% KD loss
        self.beta = 0.2   # 20% feature loss
        
        # Memory
        self.use_gradient_checkpointing = True
        self.mixed_precision = "fp16"
'''
    
    with open("config.py", 'w') as f:
        f.write(config_content)
    
    logger.info("✓ Created config.py template")


# ============================================================================
# STEP 5: Training Script
# ============================================================================

def create_train_script():
    """Create training script"""
    train_script = '''#!/usr/bin/env python3
from qwen_distill import QwenDistillationConfig, QwenDistillationTrainer, TextDataset
from torch.utils.data import DataLoader
import torch

# Load config
config = QwenDistillationConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize trainer
trainer = QwenDistillationTrainer(config, device)

# Load data
with open("data/train.txt", "r") as f:
    texts = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(texts)} text samples")

# Create dataset & dataloader
dataset = TextDataset(texts, trainer.tokenizer, max_length=config.max_seq_length)
dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

# Train
trainer.train(dataloader)

print("✓ Training complete!")
print(f"Student saved to: checkpoints/student_final.pt")
'''
    
    with open("train.py", 'w') as f:
        f.write(train_script)
    
    logger.info("✓ Created train.py")


# ============================================================================
# USAGE
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", action="store_true", help="Setup environment")
    parser.add_argument("--download", action="store_true", help="Download teacher")
    parser.add_argument("--data", action="store_true", help="Prepare dataset")
    parser.add_argument("--config", action="store_true", help="Create config")
    parser.add_argument("--all", action="store_true", help="Do all steps")
    
    args = parser.parse_args()
    
    if args.setup or args.all:
        install_dependencies()
    
    if args.download or args.all:
        download_qwen_teacher()
    
    if args.data or args.all:
        prepare_dataset("wikitext", "train", "data/train.txt")
    
    if args.config or args.all:
        create_config_template()
        create_train_script()
    
    if args.all:
        logger.info("""
        ✓ Setup complete! 
        
        Next steps:
        1. Edit config.py to customize settings
        2. Run: python train.py
        3. Monitor training in logs/
        4. Evaluate student model (see eval.py)
        """)
