import torch
import math
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig
from PIL import Image
import yaml
import os

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from src.prune import FFNAccessor
from src.method import *
from src.func import *

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def split_model(model_path):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

def load_model_tokenizer(path, dev_map='cuda:0'):
    device_map = split_model(path) if dev_map is None else dev_map
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    return model, tokenizer

def load_prompt(yaml_path, prompt_type):
    prompt_path = os.path.join(yaml_path, f'{prompt_type}.yaml')
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_file = yaml.load(f, Loader=yaml.FullLoader)

    prompt = prompt_file['qa_template']['allInfo']['prompt']
    return prompt

activations = {}

def get_activation(name):
    def hook_fn(module, input, output):
        activations[name] = output.detach()
    return hook_fn

def get_weights_and_acts(accessor, model, img_info, prompt, num_layers):
    # indices = []
    activations.clear()
    response = model.chat(tokenizer, img_info, prompt, generation_config)
    acts = accessor.compute_mlp_activate(activations, num_layers)
    weights = accessor.get_weights('down')
    return weights, acts
    # importance = get_importance(importance_type, weights=weights, activations=acts)
    # indices = torch.topk(importance, k=topk).indices
    # return indices.detach().cpu().numpy()[0]

def get_static_channels(importance, topk_keep_channel, junk_ratio=0.01, junk_keep_ratio=1.0):
    """
    Select top-k important channels and supplement with junk channels.
    
    Args:
        importance (Tensor): shape (num_layers, num_channels)
        keep_channel (int): important channels to keep
        junk_ratio (float): bottom x% of channels considered junk
        junk_keep_ratio (float): within junk, proportion to keep
    
    Returns:
        final_indices (list of LongTensor): selected indices per layer
    """
    num_layers, num_channels = importance.shape
    topk = topk_keep_channel
    junk_k = int(topk_keep_channel * junk_ratio)
    junk_keep_k = int(junk_k * junk_keep_ratio)
    assert topk + junk_keep_k <= num_channels, "Selected channels exceed total channel count"

    final_indices = []

    for i in range(num_layers):
        # top-k important
        topk_idx = torch.topk(importance[i], k=(topk-junk_k), largest=True).indices

        # junk selection (from least important)
        sorted_idx = torch.argsort(importance[i], descending=False)
        junk_pool = sorted_idx[:junk_k]
        if junk_keep_k < junk_k:
            junk_idx = junk_pool[torch.randperm(junk_k)[:junk_keep_k]]
        else:
            junk_idx = junk_pool

        # combine
        combined = torch.cat([topk_idx, junk_idx])
        final_indices.append(combined)

    return final_indices

def get_dynamic_channels(importance_scores, total_keep, static_indices_list, use_gmm=True, gmm_ratio=0.5):
    """
    Select dynamic channels for all layers based on importance scores (layer-wise).

    Args:
        importance_scores (Tensor): shape (num_layers, num_channels)
        total_keep (int): number of channels to keep per layer
        static_indices_list (List[Tensor]): list of static indices for each layer
        use_gmm (bool): whether to apply GMM for dynamic part
        gmm_ratio (float): ratio of dynamic channels selected by GMM

    Returns:
        List[Tensor]: list of final indices for each layer, each of shape (total_keep,)
    """
    from sklearn.mixture import GaussianMixture
    import numpy as np

    num_layers, num_channels = importance_scores.shape
    final_indices_list = []

    for layer_idx in range(num_layers):
        scores = importance_scores[layer_idx]  # (num_channels,)
        static_indices = static_indices_list[layer_idx]
        all_indices = torch.arange(num_channels, device=scores.device)

        # Mask out static
        static_mask = torch.ones_like(scores, dtype=torch.bool)
        static_mask[static_indices] = False
        candidate_indices = all_indices[static_mask]
        candidate_scores = scores[candidate_indices]

        dynamic_keep = total_keep - static_indices.numel()
        if dynamic_keep <= 0:
            final = static_indices[:total_keep]
            final_indices_list.append(final)
            continue

        if use_gmm and candidate_scores.numel() >= 2:
            gmm_keep = int(dynamic_keep * gmm_ratio)
            topk_keep = dynamic_keep - gmm_keep

            scores_np = candidate_scores.detach().float().cpu().numpy().reshape(-1, 1)
            gmm = GaussianMixture(n_components=2).fit(scores_np)
            probs = gmm.predict_proba(scores_np)
            means = gmm.means_.flatten()
            important_comp = int(np.argmax(means))
            gmm_mask = torch.tensor(probs[:, important_comp] > 0.5)

            gmm_selected = candidate_indices[gmm_mask][:gmm_keep]

            remaining_mask = gmm_mask.logical_not()
            remaining_scores = candidate_scores[remaining_mask]
            remaining_indices = candidate_indices[remaining_mask]
            if remaining_scores.numel() >= topk_keep:
                topk_selected = remaining_indices[torch.topk(remaining_scores, k=topk_keep).indices]
            else:
                topk_selected = remaining_indices
            dynamic_indices = torch.cat([gmm_selected, topk_selected])
        else:
            # fallback: top-k only
            dynamic_indices = candidate_indices[torch.topk(candidate_scores, k=dynamic_keep).indices]

        final = torch.cat([static_indices, dynamic_indices])[:total_keep]
        final_indices_list.append(final)

    return final_indices_list

import random
def get_random_images(img_path, img_num, seed=None):
    all_images = os.listdir(img_path)
    if seed is not None:
        random.seed(seed)  
    random.shuffle(all_images)
    return all_images[:img_num]

def print_model_size(model, pruned_model):
    sparse_model_size = get_model_size(model, count_nonzero_only=True)
    print(f"sparse model has size={sparse_model_size/MiB:.2f} MiB")
    sparse_model_params = get_num_parameters(model, count_nonzero_only=True)
    print(f"sparse model has {sparse_model_params/1e9:.2f}B parameters")
    prune_model_size = get_model_size(pruned_model, count_nonzero_only=True)
    print(f"Prune model has size={prune_model_size/MiB:.2f} MiB")
    prune_model_params = get_num_parameters(pruned_model, count_nonzero_only=True)
    print(f"Prune model has {prune_model_params/1e9:.2f}B parameters")

def set_up(model_type):
    if model_type == "9B":
        model_path = '/data/zige.wang/InternVL3/Running/Full_0620_9B/checkpoint-1640'
        model, tokenizer = load_model_tokenizer(model_path)
        accessor = FFNAccessor(model, model_type="internlm2")
        num_layers = 48
    elif model_type == "2B":
        model_path = '/data/prune/src/0530Base'
        model, tokenizer = load_model_tokenizer(model_path)
        accessor = FFNAccessor(model, model_type="qwen2")
        num_layers = 28
    return model, tokenizer, accessor, num_layers
if __name__ == "__main__":
    TABLE_TYPES = [
            'tagbar'
        ]
    yaml_path = '/home/liyuan.jiang/workspace/dataset/config/table_superior'
    imag_path = '/data/Dataset/filtered'
    generation_config = dict(max_new_tokens=4096, do_sample=False)
    num_img_per_task = 20
    topk = 4096
    total_channel_to_keep = 4096
    prune_type = 'entropy'
    # model_type = '9B'
    model_type = '2B'

    model, tokenizer, accessor, num_layers = set_up(model_type)

    accessor.register_hooks(get_activation)

    all_acts, all_weights = [], []
    ic(model)
    # forward pass
    for task in TABLE_TYPES: 
        prompt = load_prompt(yaml_path, task)
        task_acts, task_weights = [], []
        cur_img_path = os.path.join(imag_path, task)
        img_list = get_random_images(cur_img_path, num_img_per_task)
        
        for img in img_list:
            if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                ic(f'{img}')
                cur_img_info = load_image(os.path.join(cur_img_path, img)).to(torch.bfloat16).cuda()
                weights, acts = get_weights_and_acts(accessor, model, cur_img_info, prompt, num_layers)
                
                # weights: list of tensors per layer, shape: (hidden_in, hidden_out)
                # acts: list of tensors per layer, shape: (1, channels)
                task_weights.append(weights)             # shape: (num_samples, num_layers, weight_shape...)
                task_acts.append(acts.squeeze(1))        # shape: (num_samples, num_layers, channels)

        # stack over sample dimension        
        all_weights.append(task_weights)    #(task, num_samples, num_layers, hidden_in, hidden_out)
        all_acts.append(task_acts)          #(task, num_samples, num_layers, hidden_in, hidden_out)

    all_importance = get_importance(type_name=prune_type, 
                                    weights=all_weights,
                                    activations=all_acts)

    static_channel_to_keep = get_static_channels(all_importance, topk_keep_channel=topk)
    final_channel_to_keep = get_dynamic_channels(all_importance, total_channel_to_keep, static_channel_to_keep)

    pruned_model = accessor.apply_mlp_prune(final_channel_to_keep)

    pruned_model.save_pretrained(f'/home/liyuan.jiang/workspace/{prune_type}_{total_channel_to_keep}')

    ic(pruned_model)
    print_model_size(model, pruned_model)


    
