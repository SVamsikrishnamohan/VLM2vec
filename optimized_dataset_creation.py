import pandas as pd
from sklearn.model_selection import train_test_split
import random
import json
import os
import numpy as np
from tqdm import tqdm
import gc
from collections import defaultdict
import multiprocessing as mp
from functools import partial
import time
import tempfile

# --- Configuration ---
FULL_DATASET_PATH = '/dbfs/mnt/ds-themes-trends/vamsikrishna/main_dataset.csv'
OUTPUT_DIR = '/dbfs/mnt/ds-themes-trends/vamsikrishna/Main_datasplits/'

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
NUM_CANDIDATES_EVAL = 100  # Reduced from 500 for faster processing

RANDOM_SEED = 42
CHUNK_SIZE = 50000  # Larger chunks for efficiency
BATCH_SIZE = 5000   # Larger batches
MAX_MEMORY_MB = 8000  # Memory limit

# Set random seeds
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Initialize variables first
dbutils = None
spark = None

# Simple and robust Databricks detection
print("🔍 Detecting environment...")

try:
    # In Databricks notebooks, try to use dbutils directly
    # This is the simplest test - just try to use it
    dbutils.fs.ls("/")  # This will work if dbutils is available
    print("✅ Databricks environment detected - using optimized DBFS operations")
    print("✅ dbutils working correctly")
    # spark should also be available
    
except NameError:
    # dbutils not defined - definitely not in Databricks notebook
    print("❌ No Databricks connection - using local file operations")
    print("   (dbutils not available)")
    dbutils = None
    spark = None
    
except Exception as e:
    # dbutils exists but doesn't work - try other methods
    print(f"⚠️ dbutils found but not working: {e}")
    try:
        # Try Databricks Connect
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.appName("VLM2VecDataset").getOrCreate()
        from pyspark.dbutils import DBUtils  
        dbutils = DBUtils(spark)
        print("✅ Remote Databricks connection detected")
    except Exception as e2:
        print(f"❌ No Databricks connection - using local file operations")
        print(f"   (Reason: {e2})")
        dbutils = None
        spark = None


class OptimizedFileWriter:
    """Highly optimized file writer for DBFS"""
    
    def __init__(self, base_path):
        self.base_path = base_path
        self.temp_files = []
        self.batch_count = 0
        self.use_dbfs = False  # Will be set by caller
        
    def write_batch(self, data_batch, split_name):
        """Write batch to temporary file"""
        self.batch_count += 1
        
        # Create temporary file
        temp_filename = f"temp_{split_name}_{self.batch_count:06d}.jsonl"
        
        if self.use_dbfs and dbutils:
            # DBFS mode: write to local temp then copy to DBFS
            temp_path = os.path.join("/tmp", temp_filename)
            
            # Write to local temp file first (much faster)
            with open(temp_path, 'w') as f:
                for entry in data_batch:
                    f.write(json.dumps(entry) + '\n')
            
            # Copy to DBFS temp location
            dbfs_temp_path = f"/tmp/vlm2vec_temp/{temp_filename}"
            with open(temp_path, 'r') as f:
                content = f.read()
            dbutils.fs.put(dbfs_temp_path, content, overwrite=True)
            
            # Clean up local temp
            os.remove(temp_path)
            self.temp_files.append(dbfs_temp_path)
        else:
            # Local mode: write directly to output directory
            local_temp_path = os.path.join(self.base_path, temp_filename)
            with open(local_temp_path, 'w') as f:
                for entry in data_batch:
                    f.write(json.dumps(entry) + '\n')
            self.temp_files.append(local_temp_path)
        
        return len(data_batch)
    
    def finalize(self, split_name):
        """Combine all temporary files into final file"""
        final_path = os.path.join(self.base_path, f"{split_name}_dataset.jsonl")
        
        print(f"📝 Combining {len(self.temp_files)} temp files into {final_path}")
        
        if self.use_dbfs and dbutils and self.temp_files:
            # DBFS mode: read from DBFS temp files and combine
            all_content = ""
            for temp_file in tqdm(self.temp_files, desc="Combining DBFS files"):
                try:
                    content = dbutils.fs.head(temp_file, max_bytes=1024*1024*100)  # 100MB limit per read
                    all_content += content
                except Exception as e:
                    print(f"Warning: Could not read {temp_file}: {e}")
            
            # Write final combined file to DBFS
            dbutils.fs.put(final_path, all_content, overwrite=True)
            
            # Cleanup temp files
            for temp_file in self.temp_files:
                try:
                    dbutils.fs.rm(temp_file)
                except:
                    pass
        
        elif self.temp_files:
            # Local mode: combine local temp files
            with open(final_path, 'w') as final_file:
                for temp_file in tqdm(self.temp_files, desc="Combining local files"):
                    try:
                        with open(temp_file, 'r') as f:
                            final_file.write(f.read())
                    except Exception as e:
                        print(f"Warning: Could not read {temp_file}: {e}")
            
            # Cleanup temp files
            for temp_file in self.temp_files:
                try:
                    os.remove(temp_file)
                except:
                    pass
        
        self.temp_files = []
        print(f"✅ Finalized {final_path}")


def load_dataset_smart(csv_path, sample_size=None):
    """Smart dataset loading with optional sampling"""
    print(f"🔍 Loading dataset from {csv_path}...")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    
    # First, get dataset info
    chunk_iter = pd.read_csv(csv_path, chunksize=10000)
    first_chunk = next(chunk_iter)
    print(f"Dataset columns: {list(first_chunk.columns)}")
    print(f"Sample categories: {first_chunk['master_category'].value_counts()}")
    
    # Load full dataset or sample
    if sample_size:
        print(f"📊 Loading sample of {sample_size} rows for testing...")
        df = pd.read_csv(csv_path, nrows=sample_size)
    else:
        print("📊 Loading full dataset...")
        df = pd.read_csv(csv_path)
    
    # Filter and clean
    print(f"Raw dataset size: {len(df)}")
    
    # Remove rows with missing critical fields
    df = df.dropna(subset=['description', 'default_image', 'master_category'])
    
    # Filter categories with sufficient samples
    category_counts = df['master_category'].value_counts()
    valid_categories = category_counts[category_counts >= 100].index
    df = df[df['master_category'].isin(valid_categories)]
    
    print(f"Filtered dataset size: {len(df)}")
    print(f"Valid categories: {len(valid_categories)}")
    
    unique_images = df['default_image'].unique()
    print(f"Unique images: {len(unique_images)}")
    
    return df, unique_images


class FastDistractorSampler:
    """Ultra-optimized distractor sampling for large datasets"""
    
    def __init__(self, unique_images, num_pools=10):
        print(f"🎯 Creating optimized distractor sampler for {len(unique_images):,} images...")
        
        self.unique_images = np.array(unique_images)
        self.total_images = len(unique_images)
        
        # For very large datasets, use more efficient approach
        if self.total_images > 1000000:  # 1M+ images
            print("📈 Large dataset detected - using optimized sampling strategy")
            self.use_index_sampling = True
            self.max_pools = min(num_pools, 5)  # Limit pools for memory
        else:
            self.use_index_sampling = False
            self.max_pools = num_pools
            
        self.pools = []
        
        print(f"🎯 Creating {self.max_pools} distractor pools...")
        for i in range(self.max_pools):
            if self.use_index_sampling:
                # Store indices instead of full arrays for large datasets
                indices = np.arange(self.total_images)
                np.random.shuffle(indices)
                self.pools.append(indices)
            else:
                shuffled = self.unique_images.copy()
                np.random.shuffle(shuffled)
                self.pools.append(shuffled)
    
    def sample(self, ground_truth, num_needed):
        """Ultra-fast distractor sampling optimized for large datasets"""
        if self.use_index_sampling:
            return self._sample_by_index(ground_truth, num_needed)
        else:
            return self._sample_by_array(ground_truth, num_needed)
    
    def _sample_by_index(self, ground_truth, num_needed):
        """Index-based sampling for large datasets"""
        # Use random pool of indices
        pool_indices = self.pools[np.random.randint(len(self.pools))]
        
        # Find ground truth index
        try:
            gt_idx = np.where(self.unique_images == ground_truth)[0][0]
        except IndexError:
            # Ground truth not found, use any images
            selected_indices = pool_indices[:num_needed]
            return self.unique_images[selected_indices].tolist()
        
        # Remove ground truth index and sample
        mask = pool_indices != gt_idx
        valid_indices = pool_indices[mask]
        
        if len(valid_indices) >= num_needed:
            selected_indices = valid_indices[:num_needed]
            return self.unique_images[selected_indices].tolist()
        else:
            # Fallback: just sample randomly
            available_indices = np.arange(self.total_images)
            available_indices = available_indices[available_indices != gt_idx]
            np.random.shuffle(available_indices)
            selected_indices = available_indices[:num_needed]
            return self.unique_images[selected_indices].tolist()
    
    def _sample_by_array(self, ground_truth, num_needed):
        """Array-based sampling for smaller datasets"""
        # Use random pool
        pool = self.pools[np.random.randint(len(self.pools))]
        
        # Remove ground truth and sample
        mask = pool != ground_truth
        valid = pool[mask]
        
        if len(valid) >= num_needed:
            return valid[:num_needed].tolist()
        else:
            # Fallback: combine pools
            all_valid = []
            for p in self.pools:
                all_valid.extend(p[p != ground_truth])
                if len(all_valid) >= num_needed:
                    break
            return list(set(all_valid))[:num_needed]


def format_training_data(df_batch):
    """Format training data batch"""
    formatted = []
    
    for _, row in df_batch.iterrows():
        category = row['master_category']
        formatted.append({
            'instruction_text': f'Represent the given {category.lower()} product description for image retrieval task.',
            'query_text': row['description'],
            'query_image_path': None,
            'target_image_path': row['default_image'],
            'task_category': 'retrieval',
            'style_id': int(row['style_id'])
        })
    
    return formatted


def format_evaluation_data(df_batch, distractor_sampler, num_candidates=100):
    """Format evaluation data batch with progress monitoring"""
    formatted = []
    
    print(f"   📝 Processing batch of {len(df_batch)} samples with {num_candidates} candidates each...")
    
    for idx, (_, row) in enumerate(df_batch.iterrows()):
        if idx % 100 == 0 and idx > 0:
            print(f"      Progress: {idx}/{len(df_batch)} samples processed...")
            
        category = row['master_category']
        ground_truth = row['default_image']
        
        # Get distractors
        distractors = distractor_sampler.sample(ground_truth, num_candidates - 1)
        
        # Create candidates
        candidates = [{'type': 'image', 'path': ground_truth}]
        candidates.extend([{'type': 'image', 'path': d} for d in distractors])
        
        # Shuffle and find ground truth index
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
    
    print(f"   ✅ Batch complete: {len(formatted)} samples formatted")
    return formatted


def process_split_optimized(df_split, writer, split_name, is_training=True, distractor_sampler=None):
    """Process dataset split with maximum efficiency"""
    print(f"🚀 Processing {split_name}: {len(df_split)} samples")
    
    total_processed = 0
    num_batches = (len(df_split) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in tqdm(range(0, len(df_split), BATCH_SIZE), 
                  desc=f"Processing {split_name}", total=num_batches):
        
        batch_df = df_split.iloc[i:i+BATCH_SIZE]
        
        if is_training:
            formatted_batch = format_training_data(batch_df)
        else:
            formatted_batch = format_evaluation_data(batch_df, distractor_sampler, NUM_CANDIDATES_EVAL)
        
        # Write batch efficiently
        batch_size = writer.write_batch(formatted_batch, split_name)
        total_processed += batch_size
        
        # Memory management
        if i % (BATCH_SIZE * 5) == 0:
            gc.collect()
    
    # Finalize the split
    writer.finalize(split_name)
    
    print(f"✅ {split_name} complete: {total_processed} samples")
    return total_processed


def main():
    """Optimized main function"""
    global dbutils, spark  # Declare as global to access module-level variables
    
    print("🚀 OPTIMIZED VLM2Vec Dataset Creation")
    print("=" * 60)
    
    start_time = time.time()
    
    # Verify environment setup and determine if we can use DBFS
    use_dbfs = False
    if dbutils:
        try:
            # Test dbutils functionality
            dbutils.fs.ls("/")
            print("✅ DBFS access confirmed")
            use_dbfs = True
        except Exception as e:
            print(f"❌ DBFS access failed: {e}")
            print("Falling back to local mode...")
            use_dbfs = False
    
    # Create output directory
    if use_dbfs:
        try:
            dbutils.fs.mkdirs(OUTPUT_DIR)
            dbutils.fs.mkdirs("/tmp/vlm2vec_temp/")
            print(f"📁 Created DBFS directories: {OUTPUT_DIR}")
        except Exception as e:
            print(f"⚠️ Warning creating DBFS directories: {e}")
            use_dbfs = False  # Fall back to local mode
    
    if not use_dbfs:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"📁 Created local directory: {OUTPUT_DIR}")
    
    # For full dataset processing - uncomment the line below and comment the sample line
    df_full, unique_images = load_dataset_smart(FULL_DATASET_PATH)  # Full dataset
    # df_full, unique_images = load_dataset_smart(FULL_DATASET_PATH, sample_size=100000)  # Sample for testing
    
    print(f"\n📊 Dataset loaded: {len(df_full)} samples")
    print(f"📊 Category distribution:")
    print(df_full['master_category'].value_counts())
    
    # Create distractor sampler
    distractor_sampler = FastDistractorSampler(unique_images, num_pools=5)
    
    # Stratified split
    print("\n🎯 Creating stratified splits...")
    try:
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
    except ValueError as e:
        print(f"⚠️ Stratification failed: {e}")
        print("Using random split instead...")
        df_train, df_temp = train_test_split(df_full, test_size=(VAL_RATIO + TEST_RATIO), random_state=RANDOM_SEED)
        df_val, df_test = train_test_split(df_temp, test_size=(TEST_RATIO / (VAL_RATIO + TEST_RATIO)), random_state=RANDOM_SEED)
    
    print(f"✅ Split sizes:")
    print(f"   Training: {len(df_train)}")
    print(f"   Validation: {len(df_val)}")
    print(f"   Test: {len(df_test)}")
    
    # Initialize optimized writer
    writer = OptimizedFileWriter(OUTPUT_DIR)
    # Pass the dbfs flag to the writer
    writer.use_dbfs = use_dbfs
    
    # Process splits efficiently
    print("\n🏃‍♂️ Processing splits...")
    
    # Training (fastest)
    train_count = process_split_optimized(df_train, writer, "train", is_training=True)
    
    # Validation
    val_count = process_split_optimized(df_val, writer, "val", is_training=False, 
                                      distractor_sampler=distractor_sampler)
    
    # Test
    test_count = process_split_optimized(df_test, writer, "test", is_training=False,
                                       distractor_sampler=distractor_sampler)
    
    # Final summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("🎉 DATASET CREATION COMPLETE!")
    print("=" * 60)
    print(f"⏱️  Total time: {elapsed/60:.1f} minutes")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"📊 Training samples: {train_count:,}")
    print(f"📊 Validation samples: {val_count:,}")
    print(f"📊 Test samples: {test_count:,}")
    print(f"📊 Total samples: {train_count + val_count + test_count:,}")
    
    # Verify files
    for split in ["train", "val", "test"]:
        filepath = os.path.join(OUTPUT_DIR, f"{split}_dataset.jsonl")
        
        if use_dbfs and dbutils:
            # DBFS file verification
            try:
                file_info = dbutils.fs.ls(filepath)
                if file_info:
                    size_mb = file_info[0].size / (1024 * 1024)
                    print(f"📄 {split}_dataset.jsonl: {size_mb:.1f} MB")
            except Exception as e:
                print(f"⚠️ Could not verify DBFS file {filepath}: {e}")
        else:
            # Local file verification
            try:
                if os.path.exists(filepath):
                    size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    print(f"📄 {split}_dataset.jsonl: {size_mb:.1f} MB")
                else:
                    print(f"⚠️ Local file not found: {filepath}")
            except Exception as e:
                print(f"⚠️ Could not verify local file {filepath}: {e}")
    
    print("\n✅ Ready for VLM2Vec training!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🧹 Cleaning up...")
        gc.collect() 