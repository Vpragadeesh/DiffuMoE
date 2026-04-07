
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
