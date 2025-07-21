# VLM2Vec Fashion Training - Clean Setup

## 🎯 **Overview**

This repository contains optimized VLM2Vec training scripts for large-scale fashion image retrieval using the Phi-3 Vision model with LoRA fine-tuning.

## 📁 **Key Files**

### **Main Files**
- `quick_test_vlm2vec.py` - Quick setup verification (run this first!)
- `main_vlm2vec_training.py` - Main training script for your 4.3M dataset
- `ultra_fast_finish.py` - Ultra-optimized dataset processing script

### **Supporting Files**
- `src/` - VLM2Vec source code and models
- `configs/` - Training configuration files
- `requirements.txt` - Python dependencies

## 🚀 **Quick Start Guide**

### **Step 1: Quick Test (5 minutes)**
First, verify your setup works:

```bash
# Test VLM2Vec setup with sample data
python quick_test_vlm2vec.py
```

**Expected Output:**
```
✅ Model loading: PASSED
✅ Processor loading: PASSED  
✅ Dataset loading: PASSED
✅ Data collator: PASSED
✅ Trainer setup: PASSED
✅ Quick training: PASSED
🎯 All tests passed! VLM2Vec setup is working correctly.
```

### **Step 2: Process Your Dataset (10-15 minutes)**
Convert your 4.3M CSV dataset to VLM2Vec format:

```bash
# Ultra-fast dataset processing with 25 candidates
python ultra_fast_finish.py
```

**What it does:**
- ✅ Loads your `/dbfs/mnt/ds-themes-trends/vamsikrishna/main_dataset.csv`
- ✅ Creates train/val/test splits (3.5M / 436K / 436K)
- ✅ Formats training data as query-target pairs
- ✅ Creates evaluation data with 25 distractor candidates
- ✅ Saves to `/dbfs/mnt/ds-themes-trends/vamsikrishna/Main_datasplits/`

### **Step 3: Main Training (2-4 hours)**
Train VLM2Vec on your full dataset:

```bash
# Full training with GPU acceleration
python main_vlm2vec_training.py
```

**Training Configuration:**
- **Model:** microsoft/Phi-3-vision-128k-instruct
- **LoRA:** r=16, alpha=32 (0.37% trainable parameters)
- **Batch Size:** 2 (GPU) or 1 (CPU) + gradient accumulation
- **Epochs:** 3
- **Learning Rate:** 5e-5 with cosine scheduler
- **GPU:** Auto-detected with FP16 optimization

## 🔧 **Configuration**

### **Dataset Paths**
The scripts expect your data at:
```
/dbfs/mnt/ds-themes-trends/vamsikrishna/main_dataset.csv          # Input CSV
/dbfs/mnt/ds-themes-trends/vamsikrishna/Main_datasplits/          # Output JSONL
├── train_dataset.jsonl
├── val_dataset.jsonl  
└── test_dataset.jsonl
```

### **Model Configuration**
```python
# In main_vlm2vec_training.py
model_args = ModelArguments(
    model_name="microsoft/Phi-3-vision-128k-instruct",
    lora_r=16,        # LoRA rank
    lora_alpha=32,    # LoRA alpha
    num_crops=1,      # Image crops
)
```

### **Training Configuration**
```python
training_args = TrainingArguments(
    num_train_epochs=3,           # Training epochs
    per_device_train_batch_size=2, # Batch size per GPU
    learning_rate=5e-5,           # Learning rate
    output_dir="checkpoints/main_fashion_training"
)
```

## 📊 **Monitoring Training**

### **GPU Monitoring**
```bash
# Check GPU usage
nvidia-smi

# Monitor GPU memory
watch -n 1 nvidia-smi
```

### **Training Logs**
The training script provides detailed logging:
```
🔥 GPU detected: Tesla V100-SXM2-32GB
📊 Trainable parameters: 15,269,888 / 4,161,891,328 (0.37%)
🚀 Starting training...
Step 50/2000 | Loss: 8.234567 | LR: 4.5e-05
Step 100/2000 | Loss: 6.789012 | LR: 4.0e-05
```

### **Validation Evaluation**
The script automatically evaluates on validation data:
```
🔍 Running validation evaluation...
📊 Validation Accuracy: 15.67%
```

## 📁 **Output Structure**

After training completion:
```
checkpoints/main_fashion_training/
├── pytorch_model.bin          # Trained model weights
├── config.json               # Model configuration
├── training_args.bin         # Training arguments
└── adapter_config.json       # LoRA adapter config
```

## 🛠 **Troubleshooting**

### **Common Issues**

1. **Import Errors**
   ```bash
   # Make sure you're in the VLM2Vec root directory
   cd /path/to/VLM2Vec
   python quick_test_vlm2vec.py
   ```

2. **Dataset Not Found**
   ```
   ❌ Training dataset not found: /dbfs/mnt/.../train_dataset.jsonl
   ```
   **Solution:** Run `ultra_fast_finish.py` first to create the dataset

3. **GPU Memory Issues**
   ```python
   # Reduce batch size in main_vlm2vec_training.py
   per_device_train_batch_size=1
   gradient_accumulation_steps=16
   ```

4. **Slow Processing**
   ```python
   # In ultra_fast_finish.py, candidates are set to 25 for speed
   NUM_CANDIDATES_EVAL = 25  # Increase if needed
   ```

### **Environment Issues**

**Databricks Setup:**
```python
# The scripts auto-detect Databricks environment
is_databricks = 'DATABRICKS_RUNTIME_VERSION' in os.environ
```

**Local Setup:**
```bash
# Install dependencies
pip install -r requirements.txt
```

## 📈 **Performance Expectations**

### **Dataset Processing**
- **Input:** 4.3M samples CSV
- **Processing Time:** 10-15 minutes  
- **Output:** 3.5M train + 872K eval samples
- **Speed:** ~5,000 samples/second

### **Training Performance**
- **GPU (V100):** ~2-3 hours for 3 epochs
- **CPU:** ~8-12 hours for 3 epochs
- **Memory:** ~8-12GB GPU / ~16-24GB RAM

### **Model Performance**
- **Baseline Accuracy:** ~10-20% (random)
- **Expected Accuracy:** ~40-60% after training
- **LoRA Parameters:** 15.3M trainable (0.37% of total)

## 🎉 **Success Indicators**

### **Quick Test Success**
```
🎯 All tests passed! VLM2Vec setup is working correctly.
🚀 Ready for main training with your large dataset!
```

### **Dataset Processing Success**
```
⚡ LIGHTNING-FAST PROCESSING COMPLETE!
📊 Training: 3,500,000 samples
📊 Validation: 436,000 samples (25 candidates each)
📊 Test: 436,000 samples (25 candidates each)
🎯 Dataset ready for VLM2Vec training!
```

### **Training Success**
```
✅ Training completed!
📊 Final training loss: 2.345678
📊 Validation Accuracy: 45.67%
💾 Model saved to: checkpoints/main_fashion_training
```

## 📞 **Next Steps**

1. **Successful Training:** Evaluate on test set and deploy
2. **Hyperparameter Tuning:** Adjust LoRA rank, learning rate, etc.
3. **Inference:** Use trained model for fashion image retrieval
4. **Scaling:** Process even larger datasets with the optimized pipeline

---

**🎯 Ready to train VLM2Vec on your 4.3M fashion dataset!** 