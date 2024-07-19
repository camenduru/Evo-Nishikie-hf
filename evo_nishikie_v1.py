import gc
import os
from typing import Dict, List, Union

from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    UNet2DConditionModel,
)
from huggingface_hub import hf_hub_download
import safetensors
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPTextModelWithProjection


# Base models
SDXL_REPO = "stabilityai/stable-diffusion-xl-base-1.0"
DPO_REPO = "mhdang/dpo-sdxl-text2image-v1"
JN_REPO = "RunDiffusion/Juggernaut-XL-v9"
JSDXL_REPO = "stabilityai/japanese-stable-diffusion-xl"

# Evo-Ukiyoe
UKIYOE_REPO = "SakanaAI/Evo-Ukiyoe-v1"

# Evo-Nishikie
NISHIKIE_REPO = "SakanaAI/Evo-Nishikie-v1"


def load_state_dict(checkpoint_file: Union[str, os.PathLike], device: str = "cpu"):
    file_extension = os.path.basename(checkpoint_file).split(".")[-1]
    if file_extension == "safetensors":
        return safetensors.torch.load_file(checkpoint_file, device=device)
    else:
        return torch.load(checkpoint_file, map_location=device)


def load_from_pretrained(
    repo_id,
    filename="diffusion_pytorch_model.fp16.safetensors",
    subfolder="unet",
    device="cuda",
) -> Dict[str, torch.Tensor]:
    return load_state_dict(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            subfolder=subfolder,
        ),
        device=device,
    )


def reshape_weight_task_tensors(task_tensors, weights):
    """
    Reshapes `weights` to match the shape of `task_tensors` by unsqueezing in the remaining dimensions.

    Args:
        task_tensors (`torch.Tensor`): The tensors that will be used to reshape `weights`.
        weights (`torch.Tensor`): The tensor to be reshaped.

    Returns:
        `torch.Tensor`: The reshaped tensor.
    """
    new_shape = weights.shape + (1,) * (task_tensors.dim() - weights.dim())
    weights = weights.view(new_shape)
    return weights


def linear(task_tensors: List[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
    """
    Merge the task tensors using `linear`.

    Args:
        task_tensors(`List[torch.Tensor]`):The task tensors to merge.
        weights (`torch.Tensor`):The weights of the task tensors.

    Returns:
        `torch.Tensor`: The merged tensor.
    """
    task_tensors = torch.stack(task_tensors, dim=0)
    # weighted task tensors
    weights = reshape_weight_task_tensors(task_tensors, weights)
    weighted_task_tensors = task_tensors * weights
    mixed_task_tensors = weighted_task_tensors.sum(dim=0)
    return mixed_task_tensors


def merge_models(task_tensors, weights):
    keys = list(task_tensors[0].keys())
    weights = torch.tensor(weights, device=task_tensors[0][keys[0]].device)
    state_dict = {}
    for key in tqdm(keys, desc="Merging"):
        w_list = []
        for i, sd in enumerate(task_tensors):
            w = sd.pop(key)
            w_list.append(w)
        new_w = linear(task_tensors=w_list, weights=weights)
        state_dict[key] = new_w
    return state_dict


def split_conv_attn(weights):
    attn_tensors = {}
    conv_tensors = {}
    for key in list(weights.keys()):
        if any(k in key for k in ["to_k", "to_q", "to_v", "to_out.0"]):
            attn_tensors[key] = weights.pop(key)
        else:
            conv_tensors[key] = weights.pop(key)
    return {"conv": conv_tensors, "attn": attn_tensors}


def load_evo_nishikie(device="cuda") -> StableDiffusionXLControlNetPipeline:
    # Load base models
    sdxl_weights = split_conv_attn(load_from_pretrained(SDXL_REPO, device=device))
    dpo_weights = split_conv_attn(
        load_from_pretrained(
            DPO_REPO, "diffusion_pytorch_model.safetensors", device=device
        )
    )
    jn_weights = split_conv_attn(load_from_pretrained(JN_REPO, device=device))
    jsdxl_weights = split_conv_attn(load_from_pretrained(JSDXL_REPO, device=device))

    # Merge base models
    tensors = [sdxl_weights, dpo_weights, jn_weights, jsdxl_weights]
    new_conv = merge_models(
        [sd["conv"] for sd in tensors],
        [
            0.15928833971605916,
            0.1032449268871776,
            0.6503217149752791,
            0.08714501842148402,
        ],
    )
    new_attn = merge_models(
        [sd["attn"] for sd in tensors],
        [
            0.1877279276437178,
            0.20014114603909822,
            0.3922685507065275,
            0.2198623756106564,
        ],
    )

    # Delete no longer needed variables to free
    del sdxl_weights, dpo_weights, jn_weights, jsdxl_weights
    gc.collect()
    if "cuda" in device:
        torch.cuda.empty_cache()

    # Instantiate UNet
    unet_config = UNet2DConditionModel.load_config(SDXL_REPO, subfolder="unet")
    unet = UNet2DConditionModel.from_config(unet_config).to(device=device)
    unet.load_state_dict({**new_conv, **new_attn})

    # Load other modules
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        JSDXL_REPO, subfolder="text_encoder", torch_dtype=torch.float16, variant="fp16",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        JSDXL_REPO, subfolder="tokenizer", use_fast=False,
    )

    # Load Evo-Nishikie weights
    controlnet = ControlNetModel.from_pretrained(
        NISHIKIE_REPO, torch_dtype=torch.float16, device=device,
    )

    # Load pipeline
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        SDXL_REPO,
        unet=unet,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        variant="fp16",
    )

    # Load Evo-Ukiyoe weights
    pipe.load_lora_weights(UKIYOE_REPO)
    pipe.fuse_lora(lora_scale=1.0)

    pipe = pipe.to(device, dtype=torch.float16)

    return pipe