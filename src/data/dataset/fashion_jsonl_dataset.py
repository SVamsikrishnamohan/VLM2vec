"""
Fashion JSONL Dataset for VLM2Vec Training

This module implements a custom dataset class for fashion image retrieval
that works with the optimized JSONL format created by our dataset preparation script.
"""

import os
import json
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from typing import List, Tuple, Dict, Any, Optional
import datasets
from datasets.features.image import image_to_bytes
import logging
import random
from pathlib import Path

from src.data.dataset.base_pair_dataset import AutoPairDataset, add_metainfo_hook, MULTIMODAL_FEATURES
from src.model.processor import VLM_IMAGE_TOKENS, PHI3V
from src.utils import print_master, print_rank

logger = logging.getLogger(__name__)

def load_image_from_path(image_path: str, timeout: int = 10) -> Optional[Image.Image]:
    """Load image from file path or URL with error handling"""
    try:
        if image_path.startswith(('http://', 'https://')):
            # Handle URL
            response = requests.get(image_path, timeout=timeout)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            # Handle local file path
            if image_path.startswith('/dbfs/'):
                # Databricks file system path
                local_path = image_path
            else:
                # Convert relative paths
                local_path = os.path.abspath(image_path)
            
            if os.path.exists(local_path):
                image = Image.open(local_path).convert('RGB')
            else:
                logger.warning(f"Image file not found: {local_path}")
                return None
        
        return image
    except Exception as e:
        logger.warning(f"Failed to load image from {image_path}: {e}")
        return None

def create_placeholder_image(size=(224, 224)) -> Image.Image:
    """Create a placeholder white image"""
    return Image.new('RGB', size, (255, 255, 255))

def process_image(image: Image.Image, resolution: str, max_dim: int = 1344) -> Image.Image:
    """Process image according to resolution requirements"""
    if image is None:
        return create_placeholder_image()
    
    if resolution == "high":
        image = image.resize((1344, 1344))
    elif resolution == "mid":
        image = image.resize((672, 672))
    elif resolution == "low":
        image = image.resize((128, 128))
    else:
        cur_max_dim = max(image.size)
        if cur_max_dim > max_dim:
            # Maintain aspect ratio
            width, height = image.size
            if width > height:
                new_width = max_dim
                new_height = int((height * max_dim) / width)
            else:
                new_height = max_dim
                new_width = int((width * max_dim) / height)
            image = image.resize((new_width, new_height))
    return image

def get_image_data(image_path: str, model_backbone: str, image_resolution: str) -> Optional[Dict[str, Any]]:
    """Get image bytes and metadata from path"""
    if not image_path:
        return None
    
    # Load image
    image = load_image_from_path(image_path)
    if image is None:
        image = create_placeholder_image()
    
    # Process image if needed
    if model_backbone != PHI3V and image_resolution:
        image = process_image(image, image_resolution)
    
    # Convert to bytes for VLM2Vec compatibility
    try:
        bytes_data = image_to_bytes(image)
        return {"bytes": bytes_data, "path": image_path}
    except Exception as e:
        logger.warning(f"Failed to convert image to bytes: {e}")
        # Return placeholder
        placeholder = create_placeholder_image()
        bytes_data = image_to_bytes(placeholder)
        return {"bytes": bytes_data, "path": image_path}

@add_metainfo_hook
def fashion_jsonl_data_prepare(batch_dict, *args, **kwargs):
    """
    Data preparation function for fashion JSONL dataset
    Converts fashion JSONL data to VLM2Vec multimodal format
    
    For training: query_text -> target_image_path
    For evaluation: query_text -> candidate_targets (with ground_truth_index)
    """
    model_backbone = kwargs.get('model_backbone', '')
    image_resolution = kwargs.get('image_resolution', 'mid')
    dataset_split = kwargs.get('dataset_split', 'train')
    
    query_text = []
    passage_text = []
    passage_image = []
    
    if dataset_split == 'train':
        # Training data: simple text->image pairs
        for i in range(len(batch_dict['instruction_text'])):
            instruction = batch_dict['instruction_text'][i]
            query = batch_dict['query_text'][i]
            target_image_path = batch_dict['target_image_path'][i]
            
            # Combine instruction and query
            full_query = f"{instruction} {query}".strip()
            query_text.append(full_query)
            
            # Passage: image (with empty text)
            passage_text.append("")
            
            # Get image data
            image_data = get_image_data(target_image_path, model_backbone, image_resolution)
            passage_image.append(image_data)
    
    else:
        # Evaluation data: handle multiple candidates
        for i in range(len(batch_dict['instruction_text'])):
            instruction = batch_dict['instruction_text'][i]
            query = batch_dict['query_text'][i]
            candidates = batch_dict['candidate_targets'][i]
            ground_truth_idx = batch_dict['ground_truth_index'][i]
            
            # Combine instruction and query
            full_query = f"{instruction} {query}".strip()
            
            # Create one entry per candidate
            for j, candidate in enumerate(candidates):
                query_text.append(full_query)
                passage_text.append("")  # Empty text for image passages
                
                # Get candidate image
                candidate_path = candidate['path']
                image_data = get_image_data(candidate_path, model_backbone, image_resolution)
                passage_image.append(image_data)
    
    return {
        'query_text': query_text,
        'passage_text': passage_text, 
        'passage_image': passage_image,
    }

@AutoPairDataset.register('fashion_jsonl')
class FashionJSONLDataset(AutoPairDataset):
    """
    Fashion JSONL dataset class that integrates with VLM2Vec framework
    
    Expected JSONL format:
    Training:
    {
        "instruction_text": "Represent the given footwear product description for image retrieval task.",
        "query_text": "Blue denim sneakers with white soles",
        "query_image_path": null,
        "target_image_path": "/path/to/image.jpg",
        "task_category": "retrieval",
        "style_id": 12345
    }
    
    Evaluation:
    {
        "instruction_text": "Represent the given footwear product description for image retrieval task.",
        "query_text": "Blue denim sneakers with white soles", 
        "query_image_path": null,
        "candidate_targets": [
            {"type": "image", "path": "/path/to/image1.jpg"},
            {"type": "image", "path": "/path/to/image2.jpg"},
            ...
        ],
        "ground_truth_index": 0,
        "task_category": "retrieval",
        "style_id": 12345
    }
    """
    
    def __init__(self, 
                 jsonl_path: str,
                 subset_name: str = 'fashion',
                 dataset_split: str = 'train',
                 num_sample_per_subset: int = None,
                 model_args=None,
                 data_args=None,
                 training_args=None,
                 **kwargs):
        
        # Direct initialization without calling super().__init__
        self._init_fashion_dataset(
            jsonl_path=jsonl_path,
            subset_name=subset_name,
            dataset_split=dataset_split,
            num_sample_per_subset=num_sample_per_subset,
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            **kwargs
        )
        
        # Load fashion data from JSONL
        if not os.path.exists(jsonl_path):
            raise ValueError(f"JSONL file not found: {jsonl_path}")
            
        print_master(f"Loading fashion dataset from {jsonl_path}")
        self.data = self._load_jsonl_data(jsonl_path)
        
        # Limit samples if specified
        if num_sample_per_subset and num_sample_per_subset < len(self.data):
            if dataset_split == 'train':
                # For training, take first N samples
                self.data = self.data[:num_sample_per_subset]
            else:
                # For eval, sample randomly to maintain diversity
                self.data = random.sample(self.data, num_sample_per_subset)
        
        print_master(f"Loaded {len(self.data)} fashion items for {dataset_split}")
        
        # Convert to HuggingFace dataset format
        self.dataset = self._create_hf_dataset()
        self.num_rows = len(self.dataset)
    
    def _load_jsonl_data(self, jsonl_path: str) -> List[Dict[str, Any]]:
        """Load data from JSONL file"""
        data = []
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        item = json.loads(line.strip())
                        data.append(item)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                        continue
        except Exception as e:
            logger.error(f"Failed to load JSONL file {jsonl_path}: {e}")
            raise
        
        if not data:
            raise ValueError(f"No valid data found in {jsonl_path}")
        
        return data
    
    def _create_hf_dataset(self) -> datasets.Dataset:
        """Create HuggingFace dataset from loaded data"""
        if self.dataset_split == 'train':
            # Training format
            dataset_dict = {
                'instruction_text': [item['instruction_text'] for item in self.data],
                'query_text': [item['query_text'] for item in self.data],
                'target_image_path': [item['target_image_path'] for item in self.data],
                'style_id': [item['style_id'] for item in self.data]
            }
        else:
            # Evaluation format
            dataset_dict = {
                'instruction_text': [item['instruction_text'] for item in self.data],
                'query_text': [item['query_text'] for item in self.data], 
                'candidate_targets': [item['candidate_targets'] for item in self.data],
                'ground_truth_index': [item['ground_truth_index'] for item in self.data],
                'style_id': [item['style_id'] for item in self.data]
            }
        
        return datasets.Dataset.from_dict(dataset_dict)
    
    def main(self):
        """Main method required by AutoPairDataset"""
        # Apply data preparation with VLM2Vec format
        processed_dataset = self.dataset.map(
            fashion_jsonl_data_prepare,
            batched=True,
            batch_size=100,
            fn_kwargs={
                'model_backbone': getattr(self.model_args, 'model_backbone', ''),
                'image_resolution': getattr(self.data_args, 'image_resolution', 'mid'),
                'dataset_split': self.dataset_split,
                'global_dataset_name': self.subset_name
            }
        )
        
        return processed_dataset 
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Required by AutoPairDataset base class"""
        # Create instance bypassing __init__ restriction
        instance = cls.__new__(cls)
        # Call our actual initialization
        instance._init_fashion_dataset(*args, **kwargs)
        return instance
    
    @classmethod  
    def from_config(cls, config, *args, **kwargs):
        """Required by AutoPairDataset base class"""
        return cls.from_pretrained(*args, **config, **kwargs)
    
    def _init_fashion_dataset(self, 
                             jsonl_path: str, 
                             subset_name: str = 'fashion_train',
                             dataset_split: str = 'train',
                             num_sample_per_subset: Optional[int] = None,
                             model_args=None, 
                             data_args=None, 
                             training_args=None,
                             **kwargs):
        """Actual initialization logic"""
        self.jsonl_path = jsonl_path
        self.subset_name = subset_name  
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.dataset_split = dataset_split
        
        # Load fashion data from JSONL
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
            
        # Load and prepare data
        self.data = []
        with open(jsonl_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    self.data.append(item)
                    if num_sample_per_subset and len(self.data) >= num_sample_per_subset:
                        break
                except Exception as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
        
        print_master(f"✅ Loaded {len(self.data)} samples from {jsonl_path}")
        
        # Create the dataset  
        self.dataset = self._create_hf_dataset()
        self.num_rows = len(self.dataset)
    
    def __len__(self):
        """Return the number of samples in the dataset for PyTorch DataLoader compatibility"""
        return self.num_rows
    
    def __getitem__(self, idx):
        """Get item by index for PyTorch DataLoader compatibility"""
        # Return the processed dataset item
        processed_dataset = self.main()
        return processed_dataset[idx]
    
    @property
    def column_names(self):
        """Return column names for HuggingFace Transformers compatibility"""
        return ['instruction_text', 'query_text', 'query_image_path', 'target_image_path', 'task_category', 'style_id']
