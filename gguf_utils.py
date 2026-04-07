#!/usr/bin/env python3
"""
Utilities for working with GGUF models (Qwen, Mistral)
Plus comparison between GGUF teacher and student model
"""

import torch
import logging
from pathlib import Path
from typing import Optional, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# GGUF Loading (for inference only)
# ============================================================================

class GGUFWrapper:
    """
    Wrapper for loading and using GGUF models
    
    GGUF models are optimized for CPU/inference via llama.cpp
    They cannot be used for training (no gradient computation)
    
    Use cases:
    - Inference speed benchmarking
    - Comparing outputs with student model
    - Validation without loading full model into GPU
    """
    
    def __init__(self, gguf_path: str, n_gpu_layers: int = -1):
        """
        Load GGUF model
        
        Args:
            gguf_path: Path to .gguf file
            n_gpu_layers: Number of layers on GPU (-1 = all)
        """
        try:
            from llama_cpp import Llama
        except ImportError:
            logger.error("llama-cpp-python not installed. Install with:")
            logger.error("  pip install llama-cpp-python")
            raise
        
        logger.info(f"Loading GGUF: {gguf_path}")
        self.model = Llama(
            model_path=gguf_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=512,
            verbose=False,
        )
        self.gguf_path = gguf_path
        logger.info("✓ GGUF model loaded")
    
    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> str:
        """Generate text"""
        output = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            stop=["<|endoftext|>", "<|end|>"],
        )
        return output['choices'][0]['text']
    
    def get_embedding(self, text: str):
        """Get text embedding"""
        embedding = self.model.embed(text)
        return torch.tensor(embedding)
    
    def speed_test(self, prompt: str = "The future of AI", num_runs: int = 5) -> Dict:
        """Benchmark inference speed"""
        import time
        
        logger.info(f"Speed test ({num_runs} runs)...")
        times = []
        
        for _ in range(num_runs):
            start = time.time()
            self.generate(prompt, max_tokens=100)
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        logger.info(f"Average time per generation: {avg_time:.2f}s")
        logger.info(f"Throughput: {100/avg_time:.1f} tokens/sec")
        
        return {
            'avg_time_sec': avg_time,
            'throughput_tokens_per_sec': 100 / avg_time,
        }


# ============================================================================
# GGUF vs Student Comparison
# ============================================================================

class ModelComparison:
    """Compare GGUF teacher with student model"""
    
    def __init__(self, gguf_path: str, student_checkpoint: str, device: str = "cuda"):
        """
        Load both models for comparison
        
        Args:
            gguf_path: Path to GGUF teacher
            student_checkpoint: Path to student checkpoint
            device: Device for student model
        """
        self.device = torch.device(device)
        
        # Load GGUF teacher
        try:
            self.gguf_teacher = GGUFWrapper(gguf_path)
        except Exception as e:
            logger.warning(f"Could not load GGUF: {e}")
            self.gguf_teacher = None
        
        # Load student
        from qwen_inference import StudentInference
        self.student = StudentInference(student_checkpoint, device=device)
        
        self.tokenizer = self.student.tokenizer
    
    def compare_generations(self, prompt: str, max_length: int = 100):
        """Generate from both models and compare"""
        logger.info(f"\nPrompt: '{prompt}'\n")
        
        # Student generation
        logger.info("Generating with student...")
        student_text = self.student.generate(prompt, max_length=max_length)
        logger.info(f"Student:\n{student_text}\n")
        
        # GGUF generation
        if self.gguf_teacher:
            logger.info("Generating with GGUF teacher...")
            teacher_text = self.gguf_teacher.generate(prompt, max_tokens=max_length)
            logger.info(f"GGUF Teacher:\n{teacher_text}\n")
        else:
            logger.warning("GGUF teacher not available")
    
    def compare_speed(self, prompt: str = "The future of AI"):
        """Compare inference speed"""
        logger.info("\nSpeed Comparison\n")
        
        # Student speed
        logger.info("Student speed test...")
        student_stats = self.student.inference_speed_test(prompt, num_runs=10)
        
        # GGUF speed
        if self.gguf_teacher:
            logger.info("\nGGUF speed test...")
            gguf_stats = self.gguf_teacher.speed_test(prompt, num_runs=5)
            
            logger.info(f"\n{'Model':<20} {'Time (ms)':<12} {'Throughput':<20}")
            logger.info("=" * 52)
            logger.info(f"{'Student':<20} {student_stats['avg_time_ms']:<12.1f} "
                       f"{student_stats['throughput']:.1f} samples/s")
            logger.info(f"{'GGUF':<20} {gguf_stats['avg_time_sec']*1000:<12.1f} "
                       f"{gguf_stats['throughput_tokens_per_sec']:.1f} tokens/s")
            
            speedup = (gguf_stats['avg_time_sec'] * 1000) / student_stats['avg_time_ms']
            logger.info(f"\nStudent is {speedup:.1f}x faster than GGUF")
        else:
            logger.warning("GGUF teacher not available for comparison")


# ============================================================================
# Model Information & Utilities
# ============================================================================

class ModelInfo:
    """Get info about models"""
    
    @staticmethod
    def print_student_info(checkpoint_path: str):
        """Print student model info"""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = checkpoint['config']
        
        logger.info(f"\nStudent Model Info:")
        logger.info(f"{'Parameter':<30} {'Value':<20}")
        logger.info("=" * 50)
        logger.info(f"{'Layers':<30} {config.get('student_num_layers', 'N/A'):<20}")
        logger.info(f"{'Hidden Dimension':<30} {config.get('student_hidden_dim', 'N/A'):<20}")
        logger.info(f"{'Num Heads':<30} {config.get('student_num_heads', 'N/A'):<20}")
        logger.info(f"{'Max Seq Length':<30} {config.get('max_seq_length', 'N/A'):<20}")
        logger.info(f"{'Temperature':<30} {config.get('temperature', 'N/A'):<20}")
        logger.info(f"{'Training Steps':<30} {checkpoint.get('global_step', 'N/A'):<20}")
        
        # Count parameters
        model_size = sum(p.numel() for p in checkpoint['model_state_dict'].values())
        logger.info(f"{'Total Parameters':<30} {model_size/1e6:.1f}M")
        logger.info(f"{'Model Size (FP32)':<30} {model_size*4/1e9:.2f}GB")
        logger.info(f"{'Model Size (FP16)':<30} {model_size*2/1e9:.2f}GB")
    
    @staticmethod
    def gguf_info(gguf_path: str):
        """Print GGUF model info"""
        try:
            from llama_cpp import Llama
            llm = Llama(model_path=gguf_path, n_gpu_layers=0)
            logger.info(f"\nGGUF Model Info:")
            logger.info(f"Path: {gguf_path}")
            logger.info(f"Size: {Path(gguf_path).stat().st_size / 1e9:.2f}GB")
            # llama.cpp doesn't expose detailed arch info easily
        except Exception as e:
            logger.error(f"Could not load GGUF: {e}")


# ============================================================================
# Conversion Utilities
# ============================================================================

class GGUFConverter:
    """
    Convert GGUF ↔ HuggingFace formats
    
    Note: Requires knowing the model architecture
    """
    
    @staticmethod
    def gguf_to_huggingface(gguf_path: str, output_dir: str, model_type: str = "llama"):
        """
        Convert GGUF to HuggingFace format
        
        Supported model_type: "llama", "mistral", "qwen"
        
        WARNING: This is complex and often requires manual config adjustment
        Easier alternative: Download HuggingFace model directly
        """
        logger.warning("GGUF conversion is complex and model-specific")
        logger.warning("Recommend: Download equivalent from HuggingFace instead")
        logger.info(f"Example: huggingface-cli download Qwen/Qwen2.5-0.5B")


# ============================================================================
# Main - Usage Examples
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", help="Path to GGUF model")
    parser.add_argument("--student", help="Path to student checkpoint")
    parser.add_argument("--compare", action="store_true", help="Compare GGUF vs student")
    parser.add_argument("--gguf-info", action="store_true", help="Print GGUF info")
    parser.add_argument("--student-info", action="store_true", help="Print student info")
    parser.add_argument("--prompt", default="The future of AI", help="Generation prompt")
    
    args = parser.parse_args()
    
    # GGUF information
    if args.gguf_info and args.gguf:
        ModelInfo.gguf_info(args.gguf)
    
    # Student information
    if args.student_info and args.student:
        ModelInfo.print_student_info(args.student)
    
    # Comparison
    if args.compare and args.gguf and args.student:
        comp = ModelComparison(args.gguf, args.student)
        comp.compare_generations(args.prompt)
        comp.compare_speed(args.prompt)
    
    # Default: Simple GGUF loading and generation
    if args.gguf and not (args.compare or args.gguf_info):
        logger.info("Loading GGUF model (inference only)...")
        gguf = GGUFWrapper(args.gguf)
        
        logger.info(f"\nPrompt: {args.prompt}")
        text = gguf.generate(args.prompt, max_tokens=100)
        logger.info(f"\nGenerated:\n{text}")
        
        logger.info("\nSpeed test...")
        stats = gguf.speed_test(args.prompt, num_runs=3)
