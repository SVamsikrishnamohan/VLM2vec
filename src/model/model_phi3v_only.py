from typing import Dict
import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers import PreTrainedModel, AutoConfig
from peft import LoraConfig, get_peft_model, PeftModel
from src.arguments import ModelArguments, TrainingArguments
from src.model.processor_phi3v_only import PHI3V, get_backbone_name, print_master
from src.model.baseline_backbone.phi3_v.modeling_phi3_v import Phi3VForCausalLM

class MMEBModel(nn.Module):
    """Simplified VLM2Vec model - Phi-3-V only"""
    
    def __init__(self, 
                 model: PreTrainedModel,
                 model_args: ModelArguments = None,
                 training_args: TrainingArguments = None):
        super().__init__()
        self.model = model
        self.model_args = model_args
        self.training_args = training_args
        self.model_backbone = PHI3V  # Always Phi-3-V

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.model.set_input_embeddings(new_embeddings)

    def enable_input_require_grads(self):
        """Enable gradients for input embeddings"""
        self.model.get_input_embeddings().weight.requires_grad_(True)

    def forward(self, **kwargs):
        return self.model(**kwargs)

    @classmethod
    def build(cls, model_args: ModelArguments, **kwargs):
        """Build Phi-3-V model"""
        config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
        model_backbone = get_backbone_name(hf_config=config)
        print_master(f'Loading backbone [{model_backbone}] from {model_args.model_name}')
        
        # Only support Phi-3-V
        if model_backbone == PHI3V:
            config._attn_implementation = "eager"
            config.padding_side = "right"
            config.use_cache = False
            base_model = Phi3VForCausalLM.from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
        else:
            raise ValueError(f"Unsupported model backbone: {model_backbone}. Only Phi-3-V is supported.")

        # Apply LoRA if specified
        if model_args.lora_config_path:
            print_master(f"Loading LoRA config from {model_args.lora_config_path}")
            lora_config = LoraConfig.from_json_file(model_args.lora_config_path)
            base_model = get_peft_model(base_model, lora_config)
            print_master(f"LoRA enabled with config: {lora_config}")

        model = cls(base_model, model_args=model_args)
        return model

    @classmethod
    def load(cls, model_args: ModelArguments, is_trainable=True, **kwargs):
        """Load Phi-3-V model"""
        model_name_or_path = model_args.checkpoint_path if model_args.checkpoint_path else model_args.model_name
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        
        if not hasattr(model_args, "model_backbone") or not model_args.model_backbone:
            model_backbone = get_backbone_name(hf_config=config)
            setattr(model_args, 'model_backbone', model_backbone)
        
        print_master(f'Loading backbone [{model_args.model_backbone}] from {model_name_or_path}')
        
        if model_args.model_backbone == PHI3V:
            config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
            config.use_cache = False
            config.padding_side = "right"
            base_model = Phi3VForCausalLM.from_pretrained(
                model_args.model_name, 
                config=config,
                torch_dtype=torch.bfloat16, 
                trust_remote_code=True
            )
            base_model.padding_side = "right"
        else:
            raise ValueError(f"Unsupported model backbone: {model_args.model_backbone}. Only Phi-3-V is supported.")

        model = cls(base_model, model_args=model_args)
        return model

    def save_pretrained(self, save_directory: str, **kwargs):
        """Save the model"""
        self.model.save_pretrained(save_directory, **kwargs)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def to(self, device):
        return self.model.to(device) 