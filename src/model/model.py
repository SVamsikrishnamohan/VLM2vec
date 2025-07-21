from typing import Dict, Union, Any, Optional
import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers.modeling_utils import PreTrainedModel
from transformers import AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model, PeftModel
from src.arguments import ModelArguments, TrainingArguments
from src.model.processor import PHI3V, get_backbone_name, print_master, backbone2model

# ONLY Phi-3-V imports
from src.model.baseline_backbone.phi3_v.modeling_phi3_v import Phi3VForCausalLM

from transformers import modeling_utils
try:
    if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or getattr(modeling_utils, "ALL_PARALLEL_STYLES", None) is None:  # type: ignore
        setattr(modeling_utils, "ALL_PARALLEL_STYLES", ["tp", "none", "colwise", 'rowwise'])  # type: ignore
except:
    pass  # Ignore if attribute doesn't exist


class MMEBModel(nn.Module):
    TRANSFORMER_CLS = AutoModelForCausalLM

    def __init__(self,
                 encoder: Union[PreTrainedModel, PeftModel, Any],
                 pooling: str = 'last',
                 normalize: bool = False,
                 temperature: float = 0.02,
                 ):
        super().__init__()
        self.config = encoder.config
        self.encoder = encoder
        self.pooling = pooling
        self.normalize = normalize
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.is_ddp = dist.is_initialized()
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def encode_input(self, input):
        """
        Encode input for Phi-3-V model only.
        """
        # ONLY Phi-3-V processing
        hidden_states = self.encoder(**input, return_dict=True, output_hidden_states=True)
        hidden_states = hidden_states.hidden_states[-1]
        pooled_output = self._pooling(hidden_states, input['attention_mask'])
        return pooled_output

    def _pooling(self, last_hidden_state, attention_mask):
        if self.pooling == 'last' or self.pooling == 'eos':
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            batch_size = last_hidden_state.shape[0]
            if left_padding:
                # Get the vectors at the last position
                reps = last_hidden_state[torch.arange(batch_size), -1, :]
            else:
                # Calculate last 1 position in the original tensor
                eos_indices = attention_mask.sum(dim=1) - 1
                # Get the vectors at the last 1 position of each attention mask
                reps = last_hidden_state[
                    torch.arange(batch_size, device=last_hidden_state.device), eos_indices]
        else:
            raise NotImplementedError
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps

    @classmethod
    def build(cls, model_args: ModelArguments, **kwargs):
        config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
        model_backbone = get_backbone_name(hf_config=config)
        print_master(f'Loading backbone [{model_backbone}] from {model_args.model_name}')
        
        # ONLY Phi-3-V support
        if model_backbone == PHI3V:
            config._attn_implementation = "eager"
            config.padding_side = "right"
            config.use_cache = False
            base_model = Phi3VForCausalLM.from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        else:
            raise ValueError(f"Unsupported model backbone: {model_backbone}. Only 'phi3_v' is supported.")

        if model_args.lora:
            print_master(f'Loading lora adapter for {model_backbone}')
            lora_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=model_args.lora_target_modules.split(','),
                lora_dropout=model_args.lora_dropout,
                init_lora_weights="gaussian",
                use_dora=True,
                inference_mode=False
            )
            lora_model = get_peft_model(base_model, lora_config)
            model = cls(
                encoder=lora_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )
        
        # Store backbone name for reference
        model.model_backbone = model_backbone  # type: ignore
        return model

    @classmethod
    def load(cls, model_args: ModelArguments, is_trainable=True, **kwargs):
        """
        Load a pre-trained Phi-3-V model.
        """
        config = AutoConfig.from_pretrained(model_args.checkpoint_path, trust_remote_code=True)
        model_backbone = get_backbone_name(hf_config=config)
        print_master(f'Loading checkpoint [{model_backbone}] from {model_args.checkpoint_path}')
        
        # ONLY Phi-3-V support
        if model_backbone == PHI3V:
            config._attn_implementation = "eager"
            config.padding_side = "right"
            config.use_cache = False
            
            if model_args.lora:
                # Load base model
                base_model = Phi3VForCausalLM.from_pretrained(
                    model_args.model_name,
                    config=config,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                )
                # Load LoRA weights
                model = PeftModel.from_pretrained(base_model, model_args.checkpoint_path)
                if is_trainable:
                    model.train()
                else:
                    model.eval()
                    
                wrapped_model = cls(
                    encoder=model,
                    pooling=model_args.pooling,
                    normalize=model_args.normalize,
                    temperature=model_args.temperature
                )
            else:
                # Load full fine-tuned model
                base_model = Phi3VForCausalLM.from_pretrained(
                    model_args.checkpoint_path,
                    config=config,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                )
                if is_trainable:
                    base_model.train()
                else:
                    base_model.eval()
                    
                wrapped_model = cls(
                    encoder=base_model,
                    pooling=model_args.pooling,
                    normalize=model_args.normalize,
                    temperature=model_args.temperature
                )
        else:
            raise ValueError(f"Unsupported model backbone: {model_backbone}. Only 'phi3_v' is supported.")
        
        # Store backbone name for reference
        wrapped_model.model_backbone = model_backbone  # type: ignore
        return wrapped_model

    def save(self, output_dir: str):
        """
        Save the model.
        """
        self.encoder.save_pretrained(output_dir)
        print_master(f'Model saved to {output_dir}')

    def forward(self, *args, **kwargs):
        """
        Forward pass for VLM2Vec contrastive training.
        
        Handles both single input encoding and contrastive training.
        """
        # Handle contrastive training mode: model(qry=queries, tgt=targets)
        if 'qry' in kwargs and 'tgt' in kwargs:
            print(f"🚀 DEBUG: Forward called in CONTRASTIVE mode with qry and tgt")
            qry_reps = self.encode_input(kwargs['qry'])
            tgt_reps = self.encode_input(kwargs['tgt'])
            loss = self._compute_contrastive_loss(qry_reps, tgt_reps)
            return loss
        
        # Handle single input encoding: model(input_dict) or model(model_input=input_dict)
        if len(args) == 1 and isinstance(args[0], dict):
            print(f"🔧 DEBUG: Forward called in ENCODING mode with single positional arg")
            return self.encode_input(args[0])
        
        if 'model_input' in kwargs:
            return self.encode_input(kwargs['model_input'])
        
        # Handle keyword-only arguments
        if len(kwargs) == 1:
            arg_value = next(iter(kwargs.values()))
            if isinstance(arg_value, dict):
                return self.encode_input(arg_value)
        
        raise ValueError("Must provide either (qry, tgt) for training or single input dict for encoding")
    
    def _compute_contrastive_loss(self, qry_reps: Tensor, tgt_reps: Tensor) -> Tensor:
        """
        Compute contrastive loss between query and target representations.
        """
        print(f"🔍 DEBUG: qry_reps.shape = {qry_reps.shape}, tgt_reps.shape = {tgt_reps.shape}")
        print(f"🔍 DEBUG: qry_reps.norm = {qry_reps.norm():.4f}, tgt_reps.norm = {tgt_reps.norm():.4f}")
        
        # Normalize representations if specified
        if self.normalize:
            qry_reps = torch.nn.functional.normalize(qry_reps, p=2, dim=-1)
            tgt_reps = torch.nn.functional.normalize(tgt_reps, p=2, dim=-1)
            print(f"🔍 DEBUG: After normalization - qry_norm = {qry_reps.norm():.4f}, tgt_norm = {tgt_reps.norm():.4f}")
        
        # Compute similarity matrix
        logits = torch.matmul(qry_reps, tgt_reps.transpose(0, 1)) / self.temperature
        print(f"🔍 DEBUG: logits.shape = {logits.shape}, temperature = {self.temperature}")
        print(f"🔍 DEBUG: logits = {logits}")
        
        # Create labels (assuming positive pairs are at same indices)
        batch_size = qry_reps.size(0)
        labels = torch.arange(batch_size, device=qry_reps.device, dtype=torch.long)
        print(f"🔍 DEBUG: labels = {labels}")
        
        # Compute cross-entropy loss
        loss = self.cross_entropy(logits, labels)
        print(f"🔍 DEBUG: computed loss = {loss.item():.6f}")
        
        return loss
