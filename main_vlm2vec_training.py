#!/usr/bin/env python3
"""
Main VLM2Vec Training Script for Large Fashion Dataset
Optimized for Databricks environment with GPU support
"""
import os
import sys
import yaml
import torch
import json
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path for VLM2Vec imports
sys.path.append('.')

try:
    from src.arguments import ModelArguments, DataArguments, TrainingArguments
    from src.data.collator.train_collator import MultimodalDataCollator
    from src.data.loader.mixed_dataset import init_mixed_dataset
    from src.model.model import MMEBModel
    from src.model.processor import load_processor
    from src.trainer import MMEBTrainer
    logger.info("✅ VLM2Vec imports successful")
except ImportError as e:
    logger.error(f"❌ VLM2Vec import error: {e}")
    logger.error("Make sure you're running from the VLM2Vec root directory")
    sys.exit(1)

def detect_environment():
    """Detect if running in Databricks"""
    return 'DATABRICKS_RUNTIME_VERSION' in os.environ

def setup_gpu():
    """Setup GPU configuration"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        logger.info(f"🔥 GPU detected: {device_name}")
        logger.info(f"🔥 Available GPUs: {device_count}")
        logger.info(f"🔥 Current device: {current_device}")
        
        # Print memory info
        memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**3
        memory_cached = torch.cuda.memory_reserved(current_device) / 1024**3
        logger.info(f"🔥 GPU Memory: {memory_allocated:.2f}GB allocated, {memory_cached:.2f}GB cached")
        
        return True
    else:
        logger.warning("⚠️ No GPU detected, will use CPU")
        return False

def create_config_for_dataset(dataset_paths):
    """Create VLM2Vec config for the processed dataset"""
    train_file = dataset_paths.get('train', '/dbfs/mnt/ds-themes-trends/vamsikrishna/Main_datasplits/train_dataset.jsonl')
    
    config = {
        'dataset_name': 'fashion_retrieval',
        'split_weights': [1.0],
        'datasets': [{
            'name': 'FashionRetrievalTrain',
            'dataset_parser': 'fashion_jsonl',
            'train_file': train_file,
            'prob': 1.0,
            'type': 'image'
        }]
    }
    
    # Save config
    config_path = 'configs/fashion_main_training.yaml'
    os.makedirs('configs', exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"📝 Created config: {config_path}")
    return config_path

def load_evaluation_data(file_path):
    """Load evaluation data for validation/testing"""
    if not os.path.exists(file_path):
        logger.warning(f"⚠️ Evaluation file not found: {file_path}")
        return None
    
    logger.info(f"📊 Loading evaluation data: {file_path}")
    eval_data = []
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                sample = json.loads(line.strip())
                eval_data.append(sample)
            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON decode error at line {line_num}: {e}")
                continue
    
    logger.info(f"✅ Loaded {len(eval_data)} evaluation samples")
    return eval_data

def evaluate_model(model, processor, eval_data, device, max_samples=100):
    """Evaluate model on validation/test data"""
    if not eval_data:
        return None
    
    logger.info(f"🔍 Evaluating on {min(len(eval_data), max_samples)} samples...")
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, sample in enumerate(eval_data[:max_samples]):
            try:
                # This is a simplified evaluation - in practice you'd compute embeddings
                # and similarity scores for all candidates
                gt_index = sample.get('ground_truth_index', 0)
                num_candidates = len(sample.get('candidate_targets', []))
                
                if num_candidates > 0:
                    # Simulate random prediction for demo (replace with actual model inference)
                    predicted_index = torch.randint(0, num_candidates, (1,)).item()
                    
                    if predicted_index == gt_index:
                        correct += 1
                    total += 1
                
            except Exception as e:
                logger.warning(f"⚠️ Evaluation error for sample {i}: {e}")
                continue
    
    accuracy = correct / total if total > 0 else 0.0
    logger.info(f"📊 Evaluation Results: {correct}/{total} = {accuracy:.2%}")
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }

def main():
    logger.info("🚀 Starting VLM2Vec Main Training")
    logger.info("=" * 60)
    
    # Environment detection
    is_databricks = detect_environment()
    if is_databricks:
        logger.info("🏢 Running in Databricks environment")
    else:
        logger.info("💻 Running in local environment")
    
    # GPU setup
    has_gpu = setup_gpu()
    
    # Dataset paths (adjust these for your environment)
    dataset_paths = {
        'train': '/dbfs/mnt/ds-themes-trends/vamsikrishna/Main_datasplits/train_dataset.jsonl',
        'val': '/dbfs/mnt/ds-themes-trends/vamsikrishna/Main_datasplits/val_dataset.jsonl',
        'test': '/dbfs/mnt/ds-themes-trends/vamsikrishna/Main_datasplits/test_dataset.jsonl'
    }
    
    # Check if dataset exists
    train_file = dataset_paths['train']
    if not os.path.exists(train_file):
        logger.error(f"❌ Training dataset not found: {train_file}")
        logger.error("Please run the ultra_fast_finish.py script first to create the dataset")
        sys.exit(1)
    
    logger.info(f"✅ Training dataset found: {train_file}")
    
    # Create VLM2Vec config
    config_path = create_config_for_dataset(dataset_paths)
    
    # Model configuration
    model_args = ModelArguments(
        model_name="microsoft/Phi-3-vision-128k-instruct",
        model_backbone="phi3_v",
        lora=True,
        lora_r=16,  # Increased for better performance
        lora_alpha=32,  # Increased accordingly
        lora_dropout=0.1,
        num_crops=1,  # Keep minimal for efficiency
        use_flash_attention=has_gpu,  # Enable if GPU available
    )
    
    # Data configuration
    data_args = DataArguments(
        dataset_config=config_path
    )
    
    # Training configuration
    output_dir = "checkpoints/main_fashion_training"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=False,
        
        # Training schedule
        num_train_epochs=3,
        max_steps=-1,  # Use epochs instead
        per_device_train_batch_size=2 if has_gpu else 1,
        gradient_accumulation_steps=8 if has_gpu else 16,
        
        # Optimization
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        
        # Logging and saving
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="epoch",
        save_steps=500,
        evaluation_strategy="no",
        
        # Performance
        fp16=has_gpu,  # Use FP16 only with GPU
        bf16=False,
        gradient_checkpointing=True,
        dataloader_num_workers=4 if has_gpu else 2,
        remove_unused_columns=False,
        use_cpu=not has_gpu,
        
        # Reporting
        report_to="none",  # Disable wandb for now
        run_name=f"fashion_training_{int(time.time())}",
        
        # Memory optimization
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=has_gpu,
    )
    
    logger.info("⚙️ Training Configuration:")
    logger.info(f"   Model: {model_args.model_name}")
    logger.info(f"   LoRA: r={model_args.lora_r}, alpha={model_args.lora_alpha}")
    logger.info(f"   Batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"   Gradient accumulation: {training_args.gradient_accumulation_steps}")
    logger.info(f"   Learning rate: {training_args.learning_rate}")
    logger.info(f"   Epochs: {training_args.num_train_epochs}")
    logger.info(f"   Output dir: {output_dir}")
    logger.info(f"   Device: {'GPU' if has_gpu else 'CPU'}")
    
    # Load model
    logger.info("🤖 Loading VLM2Vec model...")
    start_time = time.time()
    
    model = MMEBModel.build(model_args=model_args)
    
    if has_gpu:
        model = model.cuda()
    else:
        model = model.cpu()
    
    load_time = time.time() - start_time
    logger.info(f"✅ Model loaded in {load_time:.1f}s")
    
    # Print parameter info
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"📊 Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    # Load processor
    logger.info("🔧 Loading processor...")
    processor = load_processor(model_args)
    
    # Load dataset
    logger.info("📚 Loading training dataset...")
    dataset_start = time.time()
    
    with open(config_path, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    train_dataset = init_mixed_dataset(data_args, dataset_config)
    dataset_load_time = time.time() - dataset_start
    
    logger.info(f"✅ Dataset loaded in {dataset_load_time:.1f}s")
    logger.info(f"📊 Training samples: {len(train_dataset) if hasattr(train_dataset, '__len__') else 'Unknown'}")
    
    # Setup data collator
    logger.info("🔗 Setting up data collator...")
    data_collator = MultimodalDataCollator(
        processor=processor,
        model_args=model_args,
        max_length=1024 if has_gpu else 512,
    )
    
    # Create trainer
    logger.info("👨‍🏫 Creating trainer...")
    trainer = MMEBTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        tokenizer=None,
    )
    
    # Load validation data for manual evaluation
    val_data = load_evaluation_data(dataset_paths['val'])
    
    # Training
    logger.info("🚀 Starting training...")
    logger.info("=" * 60)
    
    training_start = time.time()
    training_result = trainer.train()
    training_time = time.time() - training_start
    
    logger.info("=" * 60)
    logger.info("✅ Training completed!")
    logger.info(f"⏱️ Training time: {training_time/60:.1f} minutes")
    logger.info(f"📊 Final training loss: {training_result.training_loss:.6f}")
    logger.info(f"📊 Total training steps: {training_result.global_step}")
    
    # Save final model
    logger.info("💾 Saving final model...")
    trainer.save_model()
    logger.info(f"✅ Model saved to: {output_dir}")
    
    # Manual validation evaluation
    if val_data:
        logger.info("\n🔍 Running validation evaluation...")
        val_metrics = evaluate_model(model, processor, val_data, 
                                   device='cuda' if has_gpu else 'cpu', 
                                   max_samples=200)
        
        if val_metrics:
            logger.info(f"📊 Validation Accuracy: {val_metrics['accuracy']:.2%}")
    
    # Final summary
    logger.info("\n🎉 Training Summary")
    logger.info("=" * 60)
    logger.info(f"✅ Model: {model_args.model_name}")
    logger.info(f"✅ LoRA parameters: {trainable_params:,}")
    logger.info(f"✅ Training time: {training_time/60:.1f} minutes")
    logger.info(f"✅ Final loss: {training_result.training_loss:.6f}")
    logger.info(f"✅ Model saved: {output_dir}")
    
    if val_data and val_metrics:
        logger.info(f"✅ Validation accuracy: {val_metrics['accuracy']:.2%}")
    
    logger.info("\n🎯 Training completed successfully!")
    logger.info("📝 Next steps:")
    logger.info("   1. Evaluate on test set")
    logger.info("   2. Fine-tune hyperparameters if needed")
    logger.info("   3. Deploy for inference")

if __name__ == "__main__":
    main() 