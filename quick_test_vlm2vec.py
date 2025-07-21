#!/usr/bin/env python3
"""
Quick Test Script for VLM2Vec Setup
Tests model loading, dataset processing, and basic training on sample data
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
        return True
    else:
        logger.info("💻 No GPU detected, using CPU")
        return False

def create_test_dataset():
    """Create a small test dataset for verification"""
    logger.info("📝 Creating test dataset...")
    
    # Create sample data
    test_samples = [
        {
            "instruction_text": "Represent the given apparel product description for image retrieval task.",
            "query_text": "Blue denim jacket with silver buttons",
            "query_image_path": None,
            "target_image_path": "data/images/car.jpg",  # Using existing sample image
            "task_category": "retrieval",
            "style_id": 12345
        },
        {
            "instruction_text": "Represent the given footwear product description for image retrieval task.",
            "query_text": "Black leather boots with lace-up design",
            "query_image_path": None,
            "target_image_path": "data/images/cat.jpg",  # Using existing sample image
            "task_category": "retrieval",
            "style_id": 12346
        },
        {
            "instruction_text": "Represent the given accessories product description for image retrieval task.",
            "query_text": "Silver wristwatch with leather strap",
            "query_image_path": None,
            "target_image_path": "data/images/city.jpg",  # Using existing sample image
            "task_category": "retrieval",
            "style_id": 12347
        }
    ]
    
    # Create test directory and files
    test_dir = "test_data"
    os.makedirs(test_dir, exist_ok=True)
    
    train_file = os.path.join(test_dir, "test_train.jsonl")
    
    with open(train_file, 'w') as f:
        for sample in test_samples:
            f.write(json.dumps(sample) + '\n')
    
    logger.info(f"✅ Created test dataset: {train_file} ({len(test_samples)} samples)")
    return train_file

def create_test_config(train_file):
    """Create test configuration"""
    config = {
        'dataset_name': 'test_fashion',
        'split_weights': [1.0],
        'datasets': [{
            'name': 'TestFashion',
            'dataset_parser': 'fashion_jsonl',
            'train_file': train_file,
            'prob': 1.0,
            'type': 'image'
        }]
    }
    
    # Save config
    os.makedirs('configs', exist_ok=True)
    config_path = 'configs/test_config.yaml'
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"📝 Created test config: {config_path}")
    return config_path

def run_quick_test():
    logger.info("🧪 Starting VLM2Vec Quick Test")
    logger.info("=" * 50)
    
    # Environment detection
    is_databricks = detect_environment()
    if is_databricks:
        logger.info("🏢 Running in Databricks environment")
    else:
        logger.info("💻 Running in local environment")
    
    # GPU setup
    has_gpu = setup_gpu()
    
    # Create test dataset
    train_file = create_test_dataset()
    config_path = create_test_config(train_file)
    
    # Model configuration (minimal for testing)
    model_args = ModelArguments(
        model_name="microsoft/Phi-3-vision-128k-instruct",
        model_backbone="phi3_v",
        lora=True,
        lora_r=8,  # Small for quick test
        lora_alpha=16,
        lora_dropout=0.1,
        num_crops=1,
    )
    
    # Data configuration
    data_args = DataArguments(
        dataset_config=config_path
    )
    
    # Training configuration (minimal for testing)
    training_args = TrainingArguments(
        output_dir="test_output",
        overwrite_output_dir=True,
        do_train=True,
        do_eval=False,
        
        # Minimal training for quick test
        num_train_epochs=1,
        max_steps=3,  # Only 3 steps for testing
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        
        learning_rate=1e-5,
        weight_decay=0.01,
        warmup_steps=0,
        
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="no",
        evaluation_strategy="no",
        
        fp16=False,  # Disable for stability in test
        bf16=False,
        gradient_checkpointing=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        use_cpu=not has_gpu,
        
        report_to="none",
        run_name="quick_test",
    )
    
    logger.info("⚙️ Test Configuration:")
    logger.info(f"   Model: {model_args.model_name}")
    logger.info(f"   LoRA: r={model_args.lora_r}, alpha={model_args.lora_alpha}")
    logger.info(f"   Max steps: {training_args.max_steps}")
    logger.info(f"   Device: {'GPU' if has_gpu else 'CPU'}")
    
    # Test 1: Model loading
    logger.info("\n🧪 Test 1: Model Loading")
    logger.info("-" * 30)
    
    try:
        start_time = time.time()
        model = MMEBModel.build(model_args=model_args)
        
        if has_gpu:
            model = model.cuda()
        else:
            model = model.cpu()
        
        load_time = time.time() - start_time
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        logger.info(f"✅ Model loaded successfully in {load_time:.1f}s")
        logger.info(f"📊 Trainable: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        return False
    
    # Test 2: Processor loading
    logger.info("\n🧪 Test 2: Processor Loading")
    logger.info("-" * 30)
    
    try:
        processor = load_processor(model_args)
        logger.info("✅ Processor loaded successfully")
    except Exception as e:
        logger.error(f"❌ Processor loading failed: {e}")
        return False
    
    # Test 3: Dataset loading
    logger.info("\n🧪 Test 3: Dataset Loading")
    logger.info("-" * 30)
    
    try:
        with open(config_path, 'r') as f:
            dataset_config = yaml.safe_load(f)
        
        train_dataset = init_mixed_dataset(data_args, dataset_config)
        logger.info(f"✅ Dataset loaded successfully")
        logger.info(f"📊 Dataset size: {len(train_dataset) if hasattr(train_dataset, '__len__') else 'Unknown'}")
        
    except Exception as e:
        logger.error(f"❌ Dataset loading failed: {e}")
        return False
    
    # Test 4: Data collator
    logger.info("\n🧪 Test 4: Data Collator Setup")
    logger.info("-" * 30)
    
    try:
        data_collator = MultimodalDataCollator(
            processor=processor,
            model_args=model_args,
            max_length=512,
        )
        logger.info("✅ Data collator created successfully")
    except Exception as e:
        logger.error(f"❌ Data collator creation failed: {e}")
        return False
    
    # Test 5: Trainer setup
    logger.info("\n🧪 Test 5: Trainer Setup")
    logger.info("-" * 30)
    
    try:
        trainer = MMEBTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=None,
            data_collator=data_collator,
            tokenizer=None,
        )
        logger.info("✅ Trainer created successfully")
    except Exception as e:
        logger.error(f"❌ Trainer creation failed: {e}")
        return False
    
    # Test 6: Quick training
    logger.info("\n🧪 Test 6: Quick Training (3 steps)")
    logger.info("-" * 30)
    
    try:
        logger.info("🚀 Starting quick training...")
        start_time = time.time()
        
        training_result = trainer.train()
        
        training_time = time.time() - start_time
        
        logger.info(f"✅ Training completed in {training_time:.1f}s")
        logger.info(f"📊 Final loss: {training_result.training_loss:.6f}")
        logger.info(f"📊 Steps completed: {training_result.global_step}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return False
    
    # Cleanup
    logger.info("\n🧹 Cleaning up test files...")
    try:
        os.remove(train_file)
        os.remove(config_path)
        if os.path.exists("test_output"):
            import shutil
            shutil.rmtree("test_output")
        os.rmdir("test_data")
        logger.info("✅ Cleanup completed")
    except Exception as e:
        logger.warning(f"⚠️ Cleanup warning: {e}")
    
    # Test summary
    logger.info("\n🎉 Quick Test Results")
    logger.info("=" * 50)
    logger.info("✅ Model loading: PASSED")
    logger.info("✅ Processor loading: PASSED")
    logger.info("✅ Dataset loading: PASSED")
    logger.info("✅ Data collator: PASSED")
    logger.info("✅ Trainer setup: PASSED")
    logger.info("✅ Quick training: PASSED")
    logger.info("\n🎯 All tests passed! VLM2Vec setup is working correctly.")
    logger.info("🚀 Ready for main training with your large dataset!")
    
    return True

def main():
    """Main test function"""
    success = run_quick_test()
    
    if success:
        logger.info("\n📝 Next Steps:")
        logger.info("1. Run ultra_fast_finish.py to process your 4.3M dataset")
        logger.info("2. Run main_vlm2vec_training.py for full training")
        logger.info("3. Monitor training progress and validation metrics")
        return 0
    else:
        logger.error("\n❌ Test failed! Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 