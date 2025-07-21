from itertools import repeat
from typing import Optional
from torch.jit import isinstance

import logging
from dataclasses import dataclass
from transformers.processing_utils import ProcessorMixin
from transformers import AutoProcessor, AutoTokenizer
from src.arguments import DataArguments, ModelArguments, TrainingArguments
import torch

# ONLY Phi-3-V imports
from src.model.processor import PHI3V, process_vlm_inputs_fns
from PIL import Image
import io
from src.utils import print_rank, print_master
import os

logger = logging.getLogger(__name__)

PHI_IMAGE_TOKEN_MAX_INPUT_ID = int(1e9)

def split_and_process_vlm_inputs(model_input: dict, chunk_size: int):
    assert len(model_input) == 1
    arg_key = list(model_input.keys())[0]
    arg_val = model_input[arg_key]

    length = len(arg_val)
    chunks = []
    for i in range(0, length, chunk_size):
        chunk = {arg_key: arg_val[i:i+chunk_size]}
        chunks.append(chunk)
    
    return chunks

@dataclass
class MultimodalDataCollator:
    """
    Data collator for Phi-3-V training only.
    """
    processor: ProcessorMixin
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    def __call__(self, instances: list, chunk_size: int = 8) -> tuple:
        """
        Process a batch of instances for Phi-3-V contrastive training.
        Returns (queries, targets) tuple where:
        - queries: text descriptions processed for the model
        - targets: images processed for the model
        """
        query_texts = []
        target_images = []
        
        for instance in instances:
            # Extract query text (instruction + query_text)
            instruction = instance.get('instruction_text', '')
            query_text = instance.get('query_text', '')
            full_query = f"{instruction} {query_text}".strip()
            query_texts.append(full_query)
            
            # Extract target image
            target_image_path = instance.get('target_image_path', '')
            if target_image_path and os.path.exists(target_image_path):
                try:
                    target_image = Image.open(target_image_path).convert('RGB')
                    target_images.append(target_image)
                except Exception as e:
                    logger.warning(f"Failed to load image {target_image_path}: {e}")
                    target_images.append(None)
            else:
                target_images.append(None)
        
        # Process queries (text only)
        queries = process_vlm_inputs_fns[self.model_args.model_backbone](
            {'text': query_texts, 'images': [None] * len(query_texts)}, 
            self.processor, 
            max_length=self.data_args.max_len
        )
        
        # Process targets (image only) - wrap each image in a list and provide image tokens
        target_images_formatted = [[img] if img is not None else [None] for img in target_images]
        target_texts = [f"<|image_{i+1}|>" if img is not None else "" for i, img in enumerate(target_images)]
        targets = process_vlm_inputs_fns[self.model_args.model_backbone](
            {'text': target_texts, 'images': target_images_formatted}, 
            self.processor, 
            max_length=self.data_args.max_len
        )
        
        return queries, targets

    def prepare_inputs(self, instances: list) -> dict:
        """
        Prepare inputs from instances for Phi-3-V processing.
        """
        texts = []
        images = []
        
        for instance in instances:
            # Extract text
            text = instance.get('text', '')
            texts.append(text)
            
            # Extract and process images
            visual_input = []
            
            # Handle images from instance
            if 'images' in instance and instance['images']:
                for image_data in instance['images']:
                    image = self.process_image(image_data)
                    if image is not None:
                        visual_input.append(image)
            
            # If no images, append None
            if not visual_input:
                visual_input = None
                
            images.append(visual_input)
        
        return {'text': texts, 'images': images}

    def process_image(self, image_data) -> Optional[Image.Image]:
        """
        Process a single image for Phi-3-V.
        """
        try:
            if image_data is None:
                return None
                
            # Handle different image data types
            if isinstance(image_data, str):
                # Handle file path
                if image_data.startswith(('http://', 'https://')):
                    # URL - would need requests to download
                    logger.warning(f"URL images not supported in this collator: {image_data}")
                    return None
                else:
                    # Local file path
                    image = Image.open(image_data).convert('RGB')
            elif isinstance(image_data, bytes):
                # Handle bytes
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
            elif isinstance(image_data, Image.Image):
                # Already a PIL image
                image = image_data.convert('RGB')
            else:
                logger.warning(f"Unsupported image data type: {type(image_data)}")
                return None
                
            # Image is processed as-is for Phi-3-V
                
            return image
            
        except Exception as e:
            logger.warning(f"Failed to process image: {e}")
            return None

    def combine_chunks(self, processed_chunks: list) -> dict:
        """
        Combine processed chunks back into a single batch.
        """
        if not processed_chunks:
            return {}
            
        # Get keys from first chunk
        keys = processed_chunks[0].keys()
        combined = {}
        
        for key in keys:
            if key in ['input_ids', 'attention_mask']:
                # Concatenate along batch dimension
                combined[key] = torch.cat([chunk[key] for chunk in processed_chunks], dim=0)
            elif key in ['pixel_values', 'image_sizes']:
                # Handle image data
                values = []
                for chunk in processed_chunks:
                    if chunk[key] is not None:
                        values.extend(chunk[key] if isinstance(chunk[key], list) else [chunk[key]])
                combined[key] = values if values else None
            else:
                # For other keys, extend lists
                combined[key] = []
                for chunk in processed_chunks:
                    if isinstance(chunk[key], list):
                        combined[key].extend(chunk[key])
                    else:
                        combined[key].append(chunk[key])
        
        return combined
