from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, LlavaNextConfig, AutoConfig
import torch
from PIL import Image
import requests

config = LlavaNextConfig.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
config.text_config = AutoConfig.from_pretrained("unsloth/llama-3-8b-Instruct")

from safetensors import safe_open
sd = {}
for i in range(1, 5):
    with safe_open(f"models/model-0000{i}-of-00004.safetensors", framework="pt", device="cpu") as f:
        for key in f.keys():
            sd[key] = f.get_tensor(key)

model = LlavaNextForConditionalGeneration(config)

keys = list(sd.keys())
for key in keys:
    if 'mm_projector' not in key and 'vision_tower' not in key:
        sd['language_model.' + key] = sd[key]
        del sd[key]
keys = list(sd.keys())
for key in keys:
    if 'vision_tower' in key:
        sd[key.replace('model.vision_tower.', '')] = sd[key]
        del sd[key]
sd['multi_modal_projector.linear_1.weight'] = sd['model.mm_projector.0.weight']
sd['multi_modal_projector.linear_2.weight'] = sd['model.mm_projector.2.weight']
sd['multi_modal_projector.linear_1.bias']   = sd['model.mm_projector.0.bias']
sd['multi_modal_projector.linear_2.bias']   = sd['model.mm_projector.2.bias']
del sd['model.mm_projector.0.weight']
del sd['model.mm_projector.2.weight']
del sd['model.mm_projector.0.bias']
del sd['model.mm_projector.2.bias']
sd['image_newline'] = sd['language_model.model.image_newline']
del sd['language_model.model.image_newline']
model.load_state_dict(sd)
model.save_pretrained('models/llava-llama-3-8b')
# save language model for training
model.language_model.save_pretrained('models/llava-llama-3-8b-llm')
