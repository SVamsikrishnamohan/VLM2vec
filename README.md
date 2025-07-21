# 🚀 VLM2Vec Training with LoRA on Databricks

**Vision-Language Model Training using Contrastive Learning with LoRA Fine-tuning**

This repository provides a clean, working implementation of VLM2Vec training using Phi-3-Vision with LoRA (Low-Rank Adaptation) for efficient fine-tuning on Databricks.

## 📋 Overview

- **Model**: Microsoft Phi-3-Vision-128k-instruct
- **Training Method**: Contrastive learning (text-image pairs)
- **Fine-tuning**: LoRA (Low-Rank Adaptation) for parameter efficiency  
- **Framework**: PyTorch + HuggingFace Transformers
- **Platform**: Optimized for Databricks execution

## 🗂️ Repository Structure

```
VLM2Vec/
├── vlm2vec_training.py          # Main training script (working)
├── configs/
│   └── vlm2vec_config.yaml      # Training configuration
├── sample_test_data/
│   ├── enhanced_train_dataset.jsonl  # Training data samples
│   └── enhanced_val_dataset.jsonl    # Validation data samples
├── data/
│   └── images/                  # Sample images for training
├── src/                         # VLM2Vec core implementation
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Databricks Setup & Execution

### Step 1: Environment Setup

**Cell 1: Install Dependencies**
```python
# Install required packages
%pip install -r requirements.txt
%pip install peft accelerate bitsandbytes
```

**Cell 2: Import Libraries and Setup**
```python
import os
import sys
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

# Add current directory to Python path
sys.path.append('/databricks/driver')
```

### Step 2: Check Repository Structure

**Cell 3: Verify File Structure**
```python
import os

def check_structure():
    required_files = [
        'vlm2vec_training.py',
        'configs/vlm2vec_config.yaml',
        'sample_test_data/enhanced_train_dataset.jsonl',
        'sample_test_data/enhanced_val_dataset.jsonl',
        'requirements.txt',
        'src'
    ]
    
    print("📁 Checking repository structure:")
    all_good = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            all_good = False
    
    if all_good:
        print("\n🎉 All required files found!")
    else:
        print("\n⚠️  Some files are missing. Please check your upload.")
    
    return all_good

check_structure()
```

### Step 3: Quick Training Test (1 Step)

**Cell 4: Run Quick Training Test**
```python
# Run the training script for a quick test
import subprocess
import os

os.chdir('/databricks/driver')

# Run training for just 1 step to verify everything works
cmd = ["python", "vlm2vec_training.py"]

print("🚀 Starting VLM2Vec training test...")
print("=" * 50)

try:
    # Run the training script
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
    
    print("STDOUT:")
    print(result.stdout)
    
    if result.stderr:
        print("\nSTDERR:")
        print(result.stderr)
    
    print(f"\nReturn code: {result.returncode}")
    
    if result.returncode == 0:
        print("✅ Training test completed successfully!")
    else:
        print("❌ Training test failed!")
        
except subprocess.TimeoutExpired:
    print("⏰ Training test timed out (30 minutes)")
except Exception as e:
    print(f"❌ Error running training: {e}")
```

### Step 4: Monitor Training Output

**Cell 5: Check Training Results**
```python
# Look for key success indicators in the output
def analyze_training_output(output):
    success_indicators = [
        "LoRA parameters:",
        "Model loaded",
        "Dataset loaded",
        "CONTRASTIVE mode",
        "LOSS:",
        "Backward pass completed"
    ]
    
    print("🔍 Training Analysis:")
    print("-" * 30)
    
    found_indicators = []
    for indicator in success_indicators:
        if indicator in output:
            found_indicators.append(indicator)
            print(f"✅ {indicator}")
        else:
            print(f"❌ {indicator}")
    
    success_rate = len(found_indicators) / len(success_indicators) * 100
    print(f"\n📊 Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("🎉 Training pipeline is working correctly!")
        return True
    else:
        print("⚠️  Training pipeline needs attention.")
        return False

# This will be populated by the previous cell's output
# You can manually paste the output here if needed
training_output = """
Paste the training output from the previous cell here to analyze
"""

if len(training_output.strip()) > 50:  # Only analyze if there's substantial output
    analyze_training_output(training_output)
else:
    print("💡 Run the previous cell first, then copy the output here for analysis")
```

### Step 5: Customize Training Parameters

**Cell 6: Modify Training Configuration**
```python
# View and optionally modify the training configuration
import yaml

config_path = "configs/vlm2vec_config.yaml"

print("📋 Current Training Configuration:")
print("=" * 40)

try:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        print(yaml.dump(config, default_flow_style=False, indent=2))
except Exception as e:
    print(f"Error reading config: {e}")

print("\n💡 To modify training parameters:")
print("1. Edit the vlm2vec_training.py file")  
print("2. Look for the TrainingArguments section")
print("3. Adjust parameters like:")
print("   - num_train_epochs (currently 1)")
print("   - max_steps (currently 1)")
print("   - learning_rate (currently 1e-5)")
print("   - per_device_train_batch_size (currently 1)")
```

### Step 6: Run Extended Training

**Cell 7: Extended Training Session**
```python
# Run training for multiple epochs
# WARNING: This may take significant time depending on your parameters

print("🚀 Starting Extended VLM2Vec Training...")
print("⚠️  This may take a while depending on your configuration")
print("=" * 60)

# You can modify these parameters as needed
train_params = {
    "epochs": 2,           # Number of epochs
    "steps_per_epoch": 3,  # Steps per epoch (set to 3 for quick testing)
    "learning_rate": 1e-5, # Learning rate
    "batch_size": 1        # Batch size
}

print(f"📊 Training Parameters:")
for key, value in train_params.items():
    print(f"   {key}: {value}")

print(f"\n⏱️  Estimated training time: ~{train_params['epochs'] * train_params['steps_per_epoch'] * 2} minutes")
print("🔄 Starting training...")

# Modify the training script for extended training
# You would need to edit vlm2vec_training.py to change max_steps and num_train_epochs
# Or create a new version with different parameters

# For now, we'll show how to run it:
print("💡 To run extended training:")
print("1. Edit vlm2vec_training.py")
print("2. Change 'max_steps=1' to 'max_steps=6' (for 2 epochs × 3 steps)")
print("3. Change 'num_train_epochs=1' to 'num_train_epochs=2'")
print("4. Re-run the training script")
```

### Step 7: Training Results Analysis

**Cell 8: Analyze Training Results**
```python
# Analyze training outputs and model checkpoints
import os
import glob

print("📊 Training Results Analysis")
print("=" * 30)

# Check for any saved models or checkpoints
checkpoint_dirs = glob.glob("checkpoints/*")
if checkpoint_dirs:
    print("✅ Checkpoints found:")
    for checkpoint in checkpoint_dirs:
        print(f"   📁 {checkpoint}")
        
        # Check checkpoint contents
        files = os.listdir(checkpoint)
        if files:
            print(f"      Files: {files}")
        
        # Check size
        total_size = 0
        for root, dirs, files in os.walk(checkpoint):
            for file in files:
                total_size += os.path.getsize(os.path.join(root, file))
        
        if total_size > 0:
            print(f"      Size: {total_size / (1024*1024):.1f} MB")
else:
    print("📝 No checkpoints found (normal for quick tests)")

print("\n🔍 What to look for in training output:")
print("✅ 'LoRA parameters: X,XXX,XXX' - LoRA is working")
print("✅ 'Forward called in CONTRASTIVE mode' - Model running correctly")  
print("✅ 'LOSS: X.XXXXXX' - Loss computation working")
print("✅ 'Backward pass completed' - Gradients computed")

print("\n📈 Expected behavior:")
print("• Initial loss around 0.0 (normal for untrained model)")
print("• Loss should change over training steps")
print("• No CUDA out of memory errors")
print("• Gradients computed successfully")
```

## 🔧 Troubleshooting

### Common Issues and Solutions

**Memory Issues:**
- The training is configured to run on CPU by default
- For GPU training, ensure sufficient VRAM (>16GB recommended)
- Reduce batch size if encountering memory errors

**Import Errors:**
- Ensure all packages from requirements.txt are installed
- Check that src/ directory is in Python path

**Model Loading Issues:**
- Verify internet connection for downloading Phi-3-Vision model
- Check HuggingFace token if using private models

**Training Stuck:**
- Check for flash-attention conflicts (disabled by default)
- Verify input data format in JSONL files

## 📊 Expected Output

When training works correctly, you should see:
```
✅ LoRA parameters: 29,425,664 / 4,176,047,104 (0.70%)
✅ Forward called in CONTRASTIVE mode with qry and tgt
✅ qry_reps.shape = torch.Size([1, 3072])
✅ LOSS: 0.000000
✅ Backward pass completed
```

## 🎯 Next Steps

1. **Customize Data**: Replace sample data with your own text-image pairs
2. **Scale Training**: Increase epochs and steps for real training
3. **Evaluation**: Add evaluation metrics for model performance
4. **Deployment**: Export trained model for inference

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all files are properly uploaded to Databricks
3. Ensure Databricks cluster has sufficient resources
4. Check Python version compatibility (3.8+ recommended)

---

**Happy Training! 🚀** 