#!/usr/bin/env python3
"""
ULTRA-FAST VLM2Vec Dataset Completion
Process validation/test with maximum speed - minimal candidates
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import json
import os
import numpy as np
from tqdm import tqdm
import gc
import time

# ULTRA-AGGRESSIVE Configuration
FULL_DATASET_PATH = '/dbfs/mnt/ds-themes-trends/vamsikrishna/main_dataset.csv'
OUTPUT_DIR = '/dbfs/mnt/ds-themes-trends/vamsikrishna/Main_datasplits/'
TRAIN_FILE = os.path.join(OUTPUT_DIR, 'train_dataset.jsonl')

VAL_RATIO = 0.1
TEST_RATIO = 0.1
NUM_CANDIDATES_EVAL = 25  # ULTRA-REDUCED from 500 to 25!
BATCH_SIZE = 5000  # Larger batches
RANDOM_SEED = 42

print("⚡ ULTRA-FAST VLM2Vec Dataset Completion")
print("🎯 Strategy: Minimal candidates, maximum speed")
print("=" * 60)

# Check training
if os.path.exists(TRAIN_FILE):
    print("✅ Training dataset exists - processing val/test only")
    skip_training = True
else:
    print("⚠️ Training dataset missing")
    skip_training = False

# Load dataset efficiently
print("🔍 Loading dataset...")
start_load = time.time()
df_full = pd.read_csv(FULL_DATASET_PATH)
print(f"📊 Loaded {len(df_full):,} samples in {time.time()-start_load:.1f}s")

# Quick filter
df_full = df_full.dropna(subset=['description', 'default_image', 'master_category'])
category_counts = df_full['master_category'].value_counts()
valid_categories = category_counts[category_counts >= 100].index
df_full = df_full[df_full['master_category'].isin(valid_categories)]

print(f"📊 Filtered: {len(df_full):,} samples")

# Get unique images and create ULTRA-FAST sampler
unique_images = df_full['default_image'].unique()
print(f"📊 Unique images: {len(unique_images):,}")

class LightningDistractorSampler:
    """Lightning-fast distractor sampling for massive datasets"""
    def __init__(self, unique_images):
        self.unique_images = np.array(unique_images)
        self.n_images = len(unique_images)
        print(f"⚡ Lightning sampler: {self.n_images:,} images")
        
        # Pre-shuffle indices for ultra-fast sampling
        self.pre_shuffled_indices = np.arange(self.n_images)
        np.random.shuffle(self.pre_shuffled_indices)
        self.current_position = 0
        
    def sample(self, ground_truth, num_needed):
        """Ultra-fast sampling without any searches"""
        # Simple strategy: just take next N images from pre-shuffled pool
        # This avoids all expensive operations
        
        start_pos = self.current_position
        end_pos = start_pos + num_needed + 10  # Extra buffer
        
        if end_pos >= self.n_images:
            # Wrap around
            self.current_position = 0
            start_pos = 0
            end_pos = num_needed + 10
        
        # Get candidate indices
        candidate_indices = self.pre_shuffled_indices[start_pos:end_pos]
        candidate_images = self.unique_images[candidate_indices]
        
        # Remove ground truth if present (quick check)
        if ground_truth in candidate_images:
            mask = candidate_images != ground_truth
            candidate_images = candidate_images[mask]
        
        # Update position
        self.current_position = end_pos
        
        # Return exactly what we need
        return candidate_images[:num_needed].tolist()

print("⚡ Creating lightning-fast distractor sampler...")
sampler = LightningDistractorSampler(unique_images)

# Recreate splits quickly
print("🎯 Recreating splits...")
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

df_train, df_temp = train_test_split(df_full, test_size=(VAL_RATIO + TEST_RATIO), stratify=df_full['master_category'], random_state=RANDOM_SEED)
df_val, df_test = train_test_split(df_temp, test_size=(TEST_RATIO / (VAL_RATIO + TEST_RATIO)), stratify=df_temp['master_category'], random_state=RANDOM_SEED)

print(f"✅ Splits: Train={len(df_train):,}, Val={len(df_val):,}, Test={len(df_test):,}")

def ultra_fast_eval_format(df_batch, sampler, num_candidates=25):
    """Ultra-fast evaluation formatting with minimal operations"""
    formatted = []
    
    for _, row in df_batch.iterrows():
        category = row['master_category']
        ground_truth = row['default_image']
        
        # Lightning-fast distractor sampling
        distractors = sampler.sample(ground_truth, num_candidates - 1)
        
        # Quick candidate creation
        candidates = [{'type': 'image', 'path': ground_truth}]
        candidates.extend([{'type': 'image', 'path': d} for d in distractors])
        
        # Simple shuffle
        np.random.shuffle(candidates)
        
        # Find ground truth index quickly
        gt_index = 0
        for i, c in enumerate(candidates):
            if c['path'] == ground_truth:
                gt_index = i
                break
        
        formatted.append({
            'instruction_text': f'Represent the given {category.lower()} product description for image retrieval task.',
            'query_text': row['description'],
            'query_image_path': None,
            'candidate_targets': candidates,
            'ground_truth_index': gt_index,
            'task_category': 'retrieval',
            'style_id': int(row['style_id'])
        })
    
    return formatted

def lightning_process_split(df_split, split_name):
    """Lightning-fast split processing"""
    output_path = os.path.join(OUTPUT_DIR, f"{split_name}_dataset.jsonl")
    
    print(f"⚡ Lightning processing {split_name}: {len(df_split):,} samples")
    print(f"📁 Output: {output_path}")
    
    start_time = time.time()
    total_processed = 0
    
    # Direct file writing with large batches
    with open(output_path, 'w') as f:
        num_batches = (len(df_split) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for i in tqdm(range(0, len(df_split), BATCH_SIZE), desc=f"⚡ {split_name}", total=num_batches):
            batch_df = df_split.iloc[i:i+BATCH_SIZE]
            
            # Ultra-fast formatting
            if split_name == 'train':
                formatted_batch = []
                for _, row in batch_df.iterrows():
                    formatted_batch.append({
                        'instruction_text': f'Represent the given {row["master_category"].lower()} product description for image retrieval task.',
                        'query_text': row['description'],
                        'query_image_path': None,
                        'target_image_path': row['default_image'],
                        'task_category': 'retrieval',
                        'style_id': int(row['style_id'])
                    })
            else:
                formatted_batch = ultra_fast_eval_format(batch_df, sampler, NUM_CANDIDATES_EVAL)
            
            # Direct write
            for entry in formatted_batch:
                f.write(json.dumps(entry) + '\n')
            
            total_processed += len(formatted_batch)
            
            # Quick memory cleanup
            if i % (BATCH_SIZE * 10) == 0:
                gc.collect()
    
    elapsed = time.time() - start_time
    rate = total_processed / elapsed if elapsed > 0 else 0
    print(f"⚡ {split_name} complete: {total_processed:,} samples in {elapsed:.1f}s ({rate:.0f} samples/sec)")
    return total_processed

# ULTRA-FAST PROCESSING
overall_start = time.time()

if skip_training:
    print("⏭️ Skipping training (already complete)")
    train_count = len(df_train)
else:
    train_count = lightning_process_split(df_train, "train")

# Process validation and test with lightning speed
print("\n🚀 ULTRA-FAST VALIDATION PROCESSING")
val_count = lightning_process_split(df_val, "val")

print("\n🚀 ULTRA-FAST TEST PROCESSING")  
test_count = lightning_process_split(df_test, "test")

# Final summary
total_elapsed = time.time() - overall_start
print("\n" + "⚡" * 60)
print("🎉 LIGHTNING-FAST PROCESSING COMPLETE!")
print("⚡" * 60)
print(f"⏱️  Total time: {total_elapsed/60:.1f} minutes")
print(f"📊 Processing rate: {(val_count + test_count)/total_elapsed:.0f} samples/sec")
print(f"📊 Training: {train_count:,} samples")
print(f"📊 Validation: {val_count:,} samples ({NUM_CANDIDATES_EVAL} candidates each)")
print(f"📊 Test: {test_count:,} samples ({NUM_CANDIDATES_EVAL} candidates each)")
print(f"📊 Total: {train_count + val_count + test_count:,} samples")

# Verify files
print(f"\n📁 Files created:")
for split in ["train", "val", "test"]:
    filepath = os.path.join(OUTPUT_DIR, f"{split}_dataset.jsonl")
    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"📄 {split}_dataset.jsonl: {size_mb:.1f} MB")

print(f"\n🎯 Dataset ready for VLM2Vec training!")
print(f"💡 Note: Using {NUM_CANDIDATES_EVAL} candidates for faster processing")
print(f"💡 You can increase candidates later if needed")

# Quick sample verification
val_file = os.path.join(OUTPUT_DIR, "val_dataset.jsonl")
if os.path.exists(val_file):
    print(f"\n🔍 Validation sample:")
    with open(val_file, 'r') as f:
        sample = json.loads(f.readline())
        print(f"   Candidates: {len(sample['candidate_targets'])}")
        print(f"   Ground truth index: {sample['ground_truth_index']}")
        print(f"   ✅ Format verified!") 