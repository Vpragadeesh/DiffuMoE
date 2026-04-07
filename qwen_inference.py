#!/usr/bin/env python3
"""
Inference & Evaluation for Qwen-0.8B Student Model
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from pathlib import Path
import logging
import time
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# INFERENCE
# ============================================================================

class StudentInference:
    """Run inference with distilled student model"""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        self.checkpoint = torch.load(checkpoint_path, map_location=device)
        self.config = self.checkpoint['config']
        
        # Reconstruct student model
        from qwen_distill import QwenDistillationConfig, QwenStudentModel
        
        config_obj = QwenDistillationConfig()
        for key, val in self.config.items():
            setattr(config_obj, key, val)
        
        self.model = QwenStudentModel(config_obj).to(device)
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config_obj.teacher_model_name,
            trust_remote_code=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"✓ Model loaded. Parameters: {sum(p.numel() for p in self.model.parameters())/1e6:.1f}M")
    
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> str:
        """Generate text from prompt"""
        
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.model(input_ids)
                logits = outputs['logits'][:, -1, :]
                
                # Temperature scaling
                logits = logits / temperature
                
                # Top-p sampling
                probs = F.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Remove tokens with cumulative probability > top_p
                sorted_indices_to_remove = cumsum_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[0, indices_to_remove] = -float('inf')
                
                # Sample
                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
    def inference_speed_test(self, prompt: str = "The future of AI", num_runs: int = 10):
        """Benchmark inference speed"""
        logger.info(f"Running speed test ({num_runs} iterations)...")
        
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Warmup
        with torch.no_grad():
            _ = self.model(input_ids)
        
        # Measure
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                torch.cuda.synchronize()
                start = time.time()
                _ = self.model(input_ids)
                torch.cuda.synchronize()
                times.append(time.time() - start)
        
        avg_time = sum(times) / len(times) * 1000  # ms
        logger.info(f"Average inference time: {avg_time:.1f}ms")
        logger.info(f"Throughput: {1000/avg_time:.1f} samples/sec")
        
        return {
            'avg_time_ms': avg_time,
            'throughput': 1000 / avg_time,
        }


# ============================================================================
# EVALUATION
# ============================================================================

class StudentEvaluator:
    """Evaluate student model quality"""
    
    def __init__(self, student_checkpoint: str, teacher_model_name: str, device: str = "cuda"):
        self.device = torch.device(device)
        self.student_inf = StudentInference(student_checkpoint, device)
        
        # Load teacher
        from transformers import AutoModelForCausalLM
        logger.info(f"Loading teacher: {teacher_model_name}")
        
        self.teacher = AutoModelForCausalLM.from_pretrained(
            teacher_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.teacher.eval()
        
        self.tokenizer = self.student_inf.tokenizer
    
    def compute_perplexity(self, texts: List[str], max_length: int = 256) -> float:
        """Compute perplexity on text samples"""
        total_loss = 0.0
        num_tokens = 0
        
        self.student_inf.model.eval()
        
        with torch.no_grad():
            for text in texts:
                enc = self.tokenizer(
                    text,
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)
                
                outputs = self.student_inf.model(enc['input_ids'])
                logits = outputs['logits']
                
                # Compute cross-entropy loss
                loss = F.cross_entropy(
                    logits[0, :-1, :],
                    enc['input_ids'][0, 1:],
                    reduction='mean'
                )
                
                total_loss += loss.item()
                num_tokens += enc['input_ids'].numel()
        
        perplexity = torch.exp(torch.tensor(total_loss / len(texts))).item()
        logger.info(f"Student perplexity: {perplexity:.2f}")
        return perplexity
    
    def compute_teacher_perplexity(self, texts: List[str], max_length: int = 256) -> float:
        """Compute perplexity on teacher for comparison"""
        total_loss = 0.0
        
        self.teacher.eval()
        
        with torch.no_grad():
            for text in texts:
                enc = self.tokenizer(
                    text,
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)
                
                outputs = self.teacher(enc['input_ids'], output_hidden_states=True)
                logits = outputs.logits
                
                loss = F.cross_entropy(
                    logits[0, :-1, :],
                    enc['input_ids'][0, 1:],
                    reduction='mean'
                )
                
                total_loss += loss.item()
        
        perplexity = torch.exp(torch.tensor(total_loss / len(texts))).item()
        logger.info(f"Teacher perplexity: {perplexity:.2f}")
        return perplexity
    
    def top_k_agreement(self, texts: List[str], k: int = 5) -> float:
        """Measure how well student matches teacher top-k predictions"""
        match_count = 0
        total = 0
        
        self.student_inf.model.eval()
        self.teacher.eval()
        
        with torch.no_grad():
            for text in texts:
                enc = self.tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=256,
                    truncation=True,
                ).to(self.device)
                
                student_out = self.student_inf.model(enc['input_ids'])
                student_logits = student_out['logits']
                
                teacher_out = self.teacher(enc['input_ids'])
                teacher_logits = teacher_out.logits
                
                # Top-k tokens
                _, student_topk = torch.topk(student_logits, k, dim=-1)
                _, teacher_topk = torch.topk(teacher_logits, k, dim=-1)
                
                # Count matches
                matches = (student_topk == teacher_topk).float().sum().item()
                match_count += matches
                total += student_topk.numel()
        
        agreement = match_count / total if total > 0 else 0.0
        logger.info(f"Top-{k} agreement with teacher: {agreement*100:.1f}%")
        return agreement
    
    def generate_comparison(self, prompt: str = "The future of AI", max_length: int = 100):
        """Compare student vs teacher generation"""
        logger.info(f"\nPrompt: {prompt}\n")
        
        # Student generation
        student_text = self.student_inf.generate(prompt, max_length=max_length)
        logger.info(f"Student:\n{student_text}\n")
        
        # Teacher generation
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.teacher.generate(
                input_ids,
                max_length=max_length,
                temperature=0.7,
                top_p=0.95,
            )
        teacher_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Teacher:\n{teacher_text}\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/student_final.pt", help="Student checkpoint path")
    parser.add_argument("--teacher", default="Qwen/Qwen2.5-0.5B", help="Teacher model name")
    parser.add_argument("--prompt", default="The future of artificial intelligence", help="Generation prompt")
    parser.add_argument("--speed", action="store_true", help="Run speed test")
    parser.add_argument("--eval", action="store_true", help="Run evaluation")
    
    args = parser.parse_args()
    
    # Simple generation
    logger.info("Loading student model...")
    inference = StudentInference(args.checkpoint)
    
    logger.info(f"Generating from prompt: {args.prompt}\n")
    text = inference.generate(args.prompt, max_length=100)
    print(text)
    
    if args.speed:
        logger.info("\nBenchmarking speed...")
        inference.inference_speed_test()
    
    if args.eval:
        logger.info("\nRunning evaluation...")
        evaluator = StudentEvaluator(args.checkpoint, args.teacher)
        
        # Test data
        test_texts = [
            "Artificial intelligence is transforming industries.",
            "Machine learning models require careful tuning.",
            "Distillation compresses large models efficiently.",
        ]
        
        evaluator.compute_perplexity(test_texts)
        evaluator.compute_teacher_perplexity(test_texts)
        evaluator.top_k_agreement(test_texts, k=5)
        evaluator.generate_comparison(args.prompt, max_length=100)
