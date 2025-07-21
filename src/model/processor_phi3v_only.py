import logging
import torch
import numpy as np
from src.utils import print_master

logger = logging.getLogger(__name__)

# Only Phi-3-Vision support - clean and simple
PHI3V = 'phi3_v'
PHI_IMAGE_TOKEN_MAX_INPUT_ID = int(1e9)

# Only keep what we need for Phi-3-Vision
MODEL2BACKBONE = {
    'phi3_v': PHI3V,
}
SUPPORTED_MODELS = set(MODEL2BACKBONE.keys())

VLM_IMAGE_TOKENS = {
    PHI3V: "<|image_1|>",
}

def get_backbone_name(hf_config, model_type=None, model_name_or_path=None):
    """Get backbone name for model - Phi-3-V only"""
    if hasattr(hf_config, 'model_type'):
        if hf_config.model_type in MODEL2BACKBONE:
            return MODEL2BACKBONE[hf_config.model_type]
    
    # Default to Phi-3-V for simplicity
    return PHI3V

def load_processor(model_args, data_args=None):
    """Load processor - Phi-3-V only"""
    model_name_or_path = model_args.checkpoint_path if model_args.checkpoint_path else model_args.model_name
    print_master(f'Loading Phi-3-V processor from: {model_name_or_path}')
    
    from src.model.baseline_backbone.phi3_v.processing_phi3_v import Phi3VProcessor
    processor = Phi3VProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        num_crops=getattr(model_args, 'num_crops', 16)
    )
    processor.tokenizer.padding_side = "right"
    return processor

def Phi3V_process_fn(model_inputs: dict, processor, max_length=None):
    """Process function for Phi-3-V"""
    input_ids, pixel_values, image_sizes, image_grid_thw = [], [], [], []
    texts, images = model_inputs['text'], model_inputs['images']
    image_exists = False
    
    # Process each text-image pair
    for text, image in zip(texts, images):
        if image is None:
            inputs = processor(text, None, return_tensors="np", max_length=max_length, truncation=True)
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_sizes.append(None)
            image_grid_thw.append(None)
        else:
            image_exists = True
            inputs = processor(text=text, images=[image], return_tensors="np", max_length=max_length, truncation=True)
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            pixel_values.append(inputs['pixel_values'])
            if 'image_sizes' in inputs:
                image_sizes.append(inputs['image_sizes'])
            if 'image_grid_thw' in inputs:
                image_grid_thw.append(inputs['image_grid_thw'])

    # Padding
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    
    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'texts': texts,
        'images': images,
    }
    
    if image_exists:
        inputs['pixel_values'] = pixel_values
        inputs['image_sizes'] = image_sizes
    else:
        inputs['pixel_values'] = torch.zeros(input_ids.shape[0], 1)
        inputs['image_sizes'] = torch.ones(input_ids.shape[0], 1)
    
    return inputs 