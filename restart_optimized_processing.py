#!/usr/bin/env python3
"""
Quick restart script for VLM2Vec dataset processing
Skip training (already done) and process val/test with optimizations
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

# Configuration
FULL_DATASET_PATH = '/dbfs/mnt/ds-themes-trends/vamsikrishna/main_dataset.csv'
OUTPUT_DIR = '/dbfs/mnt/ds-themes-trends/vamsikrishna/Main_datasplits/'
TRAIN_FILE = os.path.join(OUTPUT_DIR, 'train_dataset.jsonl')

VAL_RATIO = 0.1
TEST_RATIO = 0.1
NUM_CANDIDATES_EVAL = 100  # Optimized from 500
BATCH_SIZE = 2000  # Smaller batches for val/test
RANDOM_SEED = 42

print("🚀 OPTIMIZED VLM2Vec Dataset Processing - RESTART MODE")
print("=" * 60)

# Check if training is already done
if os.path.exists(TRAIN_FILE):
    print("✅ Training dataset already exists - skipping to validation/test")
    skip_training = True
else:
    print("⚠️ Training dataset not found - will process all splits")
    skip_training = False

# Load dataset
print("🔍 Loading dataset...")
df_full = pd.read_csv(FULL_DATASET_PATH)
print(f"📊 Dataset size: {len(df_full):,} samples")

# Filter and clean
df_full = df_full.dropna(subset=['description', 'default_image', 'master_category'])
category_counts = df_full['master_category'].value_counts()
valid_categories = category_counts[category_counts >= 100].index
df_full = df_full[df_full['master_category'].isin(valid_categories)]
unique_images = df_full['default_image'].unique()

print(f"📊 Filtered size: {len(df_full):,} samples")
print(f"📊 Unique images: {len(unique_images):,}")

# Create optimized distractor sampler
print("⚡ Creating ULTRA-FAST distractor sampler...")
class OptimizedDistractorSampler:
    def __init__(self, unique_images):
        self.unique_images = np.array(unique_images)
        self.total_images = len(unique_images)
        print(f"   📈 Optimized for {self.total_images:,} images")
        
        # Pre-create large random index pool
        self.index_pool = np.arange(self.total_images)
        
    def sample(self, ground_truth, num_needed):
        # Super fast random sampling without replacement
        try:
            # Find ground truth index
            gt_mask = self.unique_images == ground_truth
            gt_indices = np.where(gt_mask)[0]
            
            if len(gt_indices) > 0:
                gt_idx = gt_indices[0]
                # Sample from all other indices
                other_indices = np.delete(self.index_pool, gt_idx)
                selected_indices = np.random.choice(other_indices, size=min(num_needed, len(other_indices)), replace=False)
            else:
                # Ground truth not found, just sample randomly
                selected_indices = np.random.choice(self.index_pool, size=min(num_needed, self.total_images), replace=False)
                
            return self.unique_images[selected_indices].tolist()
        except:
            # Fallback: simple random sampling
            return np.random.choice(self.unique_images, size=min(num_needed, self.total_images), replace=False).tolist()

distractor_sampler = OptimizedDistractorSampler(unique_images)

# Recreate the same splits
print("🎯 Recreating splits...")
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

df_train, df_temp = train_test_split(
    df_full,
    test_size=(VAL_RATIO + TEST_RATIO),
    stratify=df_full['master_category'],
    random_state=RANDOM_SEED
)

df_val, df_test = train_test_split(
    df_temp,
    test_size=(TEST_RATIO / (VAL_RATIO + TEST_RATIO)),
    stratify=df_temp['master_category'],
    random_state=RANDOM_SEED
)

print(f"✅ Splits:")
print(f"   Training: {len(df_train):,}")
print(f"   Validation: {len(df_val):,}")  
print(f"   Test: {len(df_test):,}")

def format_eval_fast(df_batch, sampler, num_candidates=100):
    """Ultra-fast evaluation formatting"""
    formatted = []
    
    for _, row in df_batch.iterrows():
        category = row['master_category']
        ground_truth = row['default_image']
        
        # Get distractors quickly
        distractors = sampler.sample(ground_truth, num_candidates - 1)
        
        # Create candidates
        candidates = [{'type': 'image', 'path': ground_truth}]
        candidates.extend([{'type': 'image', 'path': d} for d in distractors])
        
        # Quick shuffle and find index
        np.random.shuffle(candidates)
        gt_index = next(i for i, c in enumerate(candidates) if c['path'] == ground_truth)
        
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

def process_split_fast(df_split, split_name):
    """Process split with maximum speed"""
    output_path = os.path.join(OUTPUT_DIR, f"{split_name}_dataset.jsonl")
    
    print(f"🚀 Processing {split_name}: {len(df_split):,} samples")
    
    # Process in batches and write directly
    with open(output_path, 'w') as f:
        total_processed = 0
        num_batches = (len(df_split) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for i in tqdm(range(0, len(df_split), BATCH_SIZE), desc=f"Processing {split_name}"):
            batch_df = df_split.iloc[i:i+BATCH_SIZE]
            
            # Format batch
            if split_name == 'train':
                # Training format (fast)
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
                # Evaluation format
                formatted_batch = format_eval_fast(batch_df, distractor_sampler, NUM_CANDIDATES_EVAL)
            
            # Write batch directly
            for entry in formatted_batch:
                f.write(json.dumps(entry) + '\n')
            
            total_processed += len(formatted_batch)
            
            # Progress update
            if i % (BATCH_SIZE * 5) == 0:
                print(f"   Progress: {total_processed:,}/{len(df_split):,} samples")
                gc.collect()
    
    print(f"✅ {split_name} complete: {total_processed:,} samples")
    return total_processed

# Process splits
start_time = time.time()

if skip_training:
    print("⏭️ Skipping training (already complete)")
    train_count = len(df_train)
else:
    train_count = process_split_fast(df_train, "train")

# Process validation and test
val_count = process_split_fast(df_val, "val")
test_count = process_split_fast(df_test, "test")

# Summary
elapsed = time.time() - start_time
print("\n" + "=" * 60)
print("🎉 OPTIMIZED DATASET PROCESSING COMPLETE!")
print("=" * 60)
print(f"⏱️  Processing time: {elapsed/60:.1f} minutes")
print(f"📊 Training samples: {train_count:,}")
print(f"📊 Validation samples: {val_count:,}")
print(f"📊 Test samples: {test_count:,}")
print(f"📊 Total samples: {train_count + val_count + test_count:,}")

# Verify files
for split in ["train", "val", "test"]:
    filepath = os.path.join(OUTPUT_DIR, f"{split}_dataset.jsonl")
    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"📄 {split}_dataset.jsonl: {size_mb:.1f} MB")

print("\n✅ Ready for VLM2Vec training!") 