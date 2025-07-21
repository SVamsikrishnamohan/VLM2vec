import logging

import PIL
from transformers.image_utils import ChannelDimension

logger = logging.getLogger(__name__)

import torch
import numpy as np
from src.utils import print_master

# ONLY Phi-3-V imports
from src.model.baseline_backbone.phi3_v.modeling_phi3_v import Phi3VForCausalLM

PHI_IMAGE_TOKEN_MAX_INPUT_ID = int(1e9)

# ONLY Phi-3-V constants
PHI3V = 'phi3_v'

# ONLY Phi-3-V model mapping
MODEL2BACKBONE = {
    'phi3_v': PHI3V,
}
SUPPORTED_MODELS = set(MODEL2BACKBONE.keys())

# ONLY Phi-3-V tokens
VLM_IMAGE_TOKENS = {
    PHI3V: "<|image_1|>",
}

VLM_VIDEO_TOKENS = {
    # Phi-3-V doesn't use video tokens
}

backbone2model = {
    PHI3V: Phi3VForCausalLM,
}


def load_processor(model_args, data_args=None):
    """
    Load Phi-3-V processor only.
    """
    model_name_or_path = model_args.checkpoint_path if model_args.checkpoint_path else model_args.model_name
    print_master(f'Loading Phi-3-V processor from: {model_name_or_path}')
    
    # ONLY Phi-3-V support
    if model_args.model_backbone == PHI3V:
        from src.model.baseline_backbone.phi3_v.processing_phi3_v import Phi3VProcessor
        processor = Phi3VProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            num_crops=model_args.num_crops
        )
        # Set tokenizer padding side (suppress type checker warning)
        processor.tokenizer.padding_side = "right"  # type: ignore
        return processor
    else:
        raise ValueError(f"Unsupported model backbone: {model_args.model_backbone}. Only 'phi3_v' is supported.")


def get_backbone_name(hf_config, model_type=None):
    if model_type is not None:
        setattr(hf_config, 'model_type', model_type)
    assert hf_config.model_type in SUPPORTED_MODELS, f"Unknown backbone name {hf_config.model_type}.Supported models are {SUPPORTED_MODELS}"
    return MODEL2BACKBONE[hf_config.model_type]


def Phi3V_process_fn(model_inputs: dict, processor, max_length=None):
    """
    Process inputs for Phi-3-V model only.
    """
    input_ids, pixel_values, image_sizes = [], [], []
    texts, visual_inputs = model_inputs['text'], model_inputs['images']
    image_exists = False
    
    # 1. iterate each pair and process (since processors do not support batch processing)
    for text, images in zip(texts, visual_inputs):
        # in theory, each batch item should contain a list of frames, but we still check for exceptions here
        # if no images as input (not likely to happen in mmeb pro cases)
        if images is None or (type(images)==list and any(i is None for i in images)):
            inputs = processor(images=None, text=text, return_tensors="pt", max_length=max_length, truncation=True)
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # in case of empty string, only BOS is included
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_sizes.append(None)
        else:
            image_exists = True
            # in theory, valid images should be a list of frames
            assert isinstance(images, list), f"images should be a list, but got {type(images)}"
            inputs = processor(images=images, text=text, return_tensors="pt", max_length=max_length, truncation=True)
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            pixel_values.append(inputs['pixel_values'])
            image_sizes.append(inputs['image_sizes'])

    # 2. padding inputs
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    inputs = {
        'input_ids': input_ids.long(),
        'attention_mask': attention_mask,
        'texts': texts,
        'images': visual_inputs,
    }
    
    # Handle pixel values and image sizes
    image_exists = any([p is not None for p in pixel_values])
    if image_exists:
        # Process pixel values - model expects 4D tensors [num_crops, channels, height, width] per image
        processed_pixel_values = []
        processed_image_sizes = []
        
        for i in range(len(pixel_values)):
            if pixel_values[i] is not None:
                # pixel_values[i] should have shape [num_crops, channels, height, width] from processor
                pv = pixel_values[i]
                print(f"DEBUG: pixel_values[{i}] original shape: {pv.shape}")
                
                # Model expects 4D tensors [num_crops, channels, height, width] per image
                if pv.dim() == 4:  # [num_crops, channels, height, width] - correct format
                    processed_pv = pv
                elif pv.dim() == 5:  # [1, num_crops, channels, height, width] - squeeze batch dim
                    processed_pv = pv.squeeze(0)  # [num_crops, channels, height, width]
                elif pv.dim() == 3:  # [channels, height, width] - missing num_crops
                    processed_pv = pv.unsqueeze(0)  # [1, channels, height, width]
                else:
                    raise ValueError(f"Unexpected pixel_values dimension: {pv.dim()}, shape: {pv.shape}")
                
                print(f"DEBUG: pixel_values[{i}] after processing: {processed_pv.shape}")
                processed_pixel_values.append(processed_pv)
                
                # Same for image_sizes - model expects 2D tensors [num_crops, 2] per image  
                is_val = image_sizes[i]
                print(f"DEBUG: image_sizes[{i}] original shape: {is_val.shape}")
                
                if is_val.dim() == 2:  # [num_crops, 2] - correct format
                    processed_is = is_val
                elif is_val.dim() == 3:  # [1, num_crops, 2] - squeeze batch dim
                    processed_is = is_val.squeeze(0)  # [num_crops, 2]
                elif is_val.dim() == 1:  # [2] - missing num_crops
                    processed_is = is_val.unsqueeze(0)  # [1, 2]
                else:
                    raise ValueError(f"Unexpected image_sizes dimension: {is_val.dim()}, shape: {is_val.shape}")
                    
                print(f"DEBUG: image_sizes[{i}] after processing: {processed_is.shape}")
                processed_image_sizes.append(processed_is)
            else:
                # Create placeholder for None images with correct 4D shape
                print(f"DEBUG: Creating placeholder for None image at index {i}")
                processed_pixel_values.append(torch.zeros(1, 3, 336, 336))  # [1, channels, height, width]
                processed_image_sizes.append(torch.ones(1, 2))  # [1, 2]
        
        # Store as list of tensors (not concatenated) - model will handle concatenation
        inputs['pixel_values'] = processed_pixel_values
        inputs['image_sizes'] = processed_image_sizes
        
        # Debug: Print final tensor shapes
        print(f"DEBUG: Final pixel_values list lengths and shapes:")
        for i, pv in enumerate(processed_pixel_values):
            print(f"  pixel_values[{i}]: {pv.shape}")
        print(f"DEBUG: Final image_sizes list lengths and shapes:")
        for i, is_val in enumerate(processed_image_sizes):
            print(f"  image_sizes[{i}]: {is_val.shape}")
            
    else:
        # No images - create None placeholders for text-only inputs
        batch_size = input_ids.shape[0]
        print(f"DEBUG: No images - creating {batch_size} None placeholders for text-only inputs")
        inputs['pixel_values'] = [None] * batch_size
        inputs['image_sizes'] = [None] * batch_size

    return inputs


def process_input_text(instruction, model_backbone, text=None, add_video_token=False, add_image_token=False):
    """
    Formulate input text for Phi-3-V only.
    """
    if model_backbone != PHI3V:
        raise ValueError(f"Unsupported model backbone: {model_backbone}. Only 'phi3_v' is supported.")
    
    prompt = instruction
    if text:
        prompt = prompt + " " + text
    
    # Phi-3-V doesn't use video tokens
    if add_video_token:
        raise ValueError("Phi-3-V does not support video tokens")
        
    if add_image_token:
        image_token = VLM_IMAGE_TOKENS[model_backbone]
        prompt = image_token + " " + prompt

    return prompt


# ONLY Phi-3-V processing function
process_vlm_inputs_fns = {
    PHI3V: Phi3V_process_fn,
}
