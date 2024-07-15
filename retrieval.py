import csv
import json
import os
import math
import glob
import random
import sys
import torch
from contextlib import suppress
from tqdm import tqdm

import argparse
from accelerate import Accelerator
import transformers
from copy import copy
from itertools import product


import logging

from collections import defaultdict
from html.parser import HTMLParser
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image
from datasets import load_from_disk, load_dataset


from torch import nn
import torch.nn.functional as F

from transformers import AutoTokenizer
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import LlavaNextConfig, AutoModel
from transformers.models.llava_next.modeling_llava_next import LlavaNextMultiModalProjector
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor

from peft import PeftModel

DEBUG = False
MODEL_TYPE = 'llava'

accelerator = Accelerator()

llama3_template = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n'

def recall_at_k(scores, positive_pairs, k):
    """
    Compute the recall at k for each sample
    :param scores: compability score between  text and image embeddings (nb texts, nb images)
    :param k: number of images to consider per text, for retrieval
    :param positive_pairs: boolean matrix of positive pairs (nb texts, nb images)
    :return: recall at k averaged over all texts
    """
    nb_texts, nb_images = scores.shape
    # for each text, sort according to image scores in decreasing order
    topk_indices = torch.topk(scores, k, dim=1)[1]
    # compute number of positives for each text
    nb_positive = positive_pairs.sum(dim=1)
    # nb_texts, k, nb_images
    topk_indices_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=nb_images)
    # compute number of true positives
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    # a true positive means a positive among the topk
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1,2))
    # compute recall at k
    recall_at_k = (nb_true_positive / nb_positive)
    return recall_at_k

def batchify(func, X, Y, batch_size, device, *args, **kwargs):
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end].to(device)
        y = Y[start:end].to(device)
        result = func(x, y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)

def emb_data(model, transform, dataset, device,
             emb_type='text', prompt=None, bsz=4,
             text_column='caption', img_column='img'):
    # emb img
    def custom_collate_fn(batch):
        collated_batch = {}
        for key in batch[0].keys():
            collated_batch[key] = [b[key] for b in batch]
        return collated_batch

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=3*bsz if emb_type == 'text' else bsz,
        shuffle=False, num_workers=1,
        collate_fn=custom_collate_fn
    )
    dataloader = accelerator.prepare(dataloader)
    embs = []
    bar = tqdm(total=len(dataloader))
    for batch in dataloader:
        if emb_type == 'text':
            input_texts = [prompt.replace('<sent>', text) for text in sum(batch[text_column], start=[])]
            inputs = transform(input_texts,
                               return_tensors="pt", padding=True)
            for key in inputs:
                if inputs[key] is not None:
                   inputs[key] = inputs[key].to(device)
        else:
            input_texts = [prompt]*len(batch[img_column])
            if MODEL_TYPE == 'phi3':
                # phi3 only support 1 bsz for image
                assert len(input_texts) == 1
                input_texts = input_texts[0]
            inputs = transform(input_texts,
                               batch[img_column], return_tensors="pt", padding=True).to(device)


        with torch.no_grad():
            emb = model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
            emb = F.normalize(emb, dim=-1)
        emb = accelerator.gather(emb)
        embs.append(emb.cpu().float())
        bar.update(1)
    embs = torch.cat(embs)
    total = 0
    for i in dataset:
        if emb_type == 'text' and type(i[text_column]) is list:
            total += len(i[text_column])
        else:
            total += 1
    bar.close()
    return embs[:total]

def log_to_file(data, metrics, checkpoint_name, fiq_data_type=None, orc_replace_text=False):
    if data == 'flickr30k' or data == 'coco':
        if orc_replace_text:
            output = f"orc {data}: {metrics['image_retrieval_recall@5']:.4f} {metrics['text_retrieval_recall@5']:.4f}"
        else:
            output = f"{data}: {metrics['image_retrieval_recall@5']:.4f} {metrics['text_retrieval_recall@5']:.4f}"
    elif data == 'fashioniq':
        assert len(metrics) == 2
        r_at_1, r_at_5 = metrics
        output = f"{data} {fiq_data_type}: R@10: {r_at_1:.4f} R@50: {r_at_5:.4f}"
    elif data == 'cirr':
        assert len(metrics) == 3
        r_at_1, r_at_3, r_at_5 = metrics
        output = f"{data}:  R@1: {r_at_1:.4f} R@5: {r_at_3:.4f} R@10: {r_at_5:.4f}"

    if checkpoint_name is not None:
        with open(checkpoint_name, 'a') as f:
            print(output, file=f)
    return output

def init_model_and_transform(lora_path, bf16, fp32, use_e5v=False):
    dtype = torch.bfloat16 if bf16 else torch.float16
    if fp32:
        dtype = torch.float32

    if MODEL_TYPE == 'phi3':
        transform = AutoProcessor.from_pretrained("microsoft/Phi-3-vision-128k-instruct", trust_remote_code=True)

        if lora_path is not None:
            merge_path = 'merged-' + lora_path.replace('/', '-').replace('.', '')
            with accelerator.main_process_first():
                if not os.path.exists(merge_path):
                    model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-vision-128k-instruct",
                                                    device_map="cuda", trust_remote_code=True,
                                                    torch_dtype=dtype, _attn_implementation='eager')
                    model = PeftModel.from_pretrained(model, lora_path).merge_and_unload()
                    model.save_pretrained(merge_path, safe_serialization=False)
            model_name = merge_path
        else:
            model_name = "microsoft/Phi-3-vision-128k-instruct"
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                    device_map="cuda", trust_remote_code=True,
                                                    torch_dtype=dtype, _attn_implementation='eager')
        transform.tokenizer.padding_side = "left"
        transform.tokenizer.padding = True
        return model, transform
    else:
        MODEL_CLASS = LlavaNextForConditionalGeneration
        transform = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        if MODEL_TYPE == 'llava_llama3':
            tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3-8b-Instruct")
            transform.tokenizer = tokenizer
            transform.tokenizer.add_tokens('<image>')
            transform.tokenizer.pad_token_id = transform.tokenizer.eos_token_id
        transform.tokenizer.padding_side = "left"
        transform.tokenizer.padding = True

    model_name = "llava-hf/llava-v1.6-mistral-7b-hf"

    if MODEL_TYPE == 'llava_llama3':
        model_name = "./llava-llama-3-8b"

    if lora_path is not None:
        merge_path = 'merged-' + lora_path.replace('/', '-').replace('.', '')
        with accelerator.main_process_first():
            if not os.path.exists(merge_path):
                model = MODEL_CLASS.from_pretrained(model_name,
                                                    device_map='cpu')
                model.language_model = PeftModel.from_pretrained(model.language_model, lora_path).merge_and_unload()
                model.save_pretrained(merge_path)
        model_name = merge_path

    if use_e5v:
        model_name = 'royokong/e5-v'
        transform = LlavaNextProcessor.from_pretrained('royokong/e5-v')


    model = MODEL_CLASS.from_pretrained(model_name,
                                        torch_dtype=dtype, low_cpu_mem_usage=True)
    if MODEL_TYPE == 'llava_llama3':
        model.config.image_token_index = 128256

    return model, transform

def create_text_image(text, image_width=800, image_height=400, font_path="arial.ttf",
                      font_size=40, background_color=(255, 255, 255), text_color=(0, 0, 0)):
    from PIL import Image, ImageDraw, ImageFont
    image = Image.new('RGB', (image_width, image_height), color=background_color)

    # Initialize ImageDraw
    draw = ImageDraw.Draw(image)

    # Load the font
    font = ImageFont.truetype(font_path, font_size)

    # Function to wrap text
    def draw_text_with_wrapping(draw, text, font, max_width):
        lines = []
        words = text.split()
        while words:
            line = ''
            while words and draw.textlength(line + words[0], font=font) <= max_width:
                line += (words.pop(0) + ' ')
            lines.append(line)
        return lines

    # Calculate the maximum width for the text
    max_text_width = image_width - 40  # Adding some padding

    # Get the lines of wrapped text
    lines = draw_text_with_wrapping(draw, text, font, max_text_width)

    # Calculate the position for the text
    total_text_height = sum(draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1] for line in lines)
    text_x = 20
    text_y = (image_height - total_text_height) // 2

    # Add text to image
    for line in lines:
        draw.text((text_x, text_y), line, font=font, fill=text_color)
        text_y += draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1]

    return image

def ir(model, transform,
       img_prompt, text_prompt,
       data, device,
       ocr_replace_text=False,
       batch_size=None):
    dataset = load_dataset(f'royokong/{data}_test', split='test')

    dataset = dataset.rename_column('text', 'caption')
    dataset = dataset.rename_column('image', 'img')
    if data == 'coco':
        dataset = dataset.map(lambda x: {'caption': x['caption'][:5]}, num_proc=4)

    bsz = 4
    if batch_size is not None:
        bsz = batch_size

    if ocr_replace_text:
        with accelerator.main_process_first():
            if os.path.exists(f'{data}_ocr'):
                ocr_dataset = load_from_disk(f'{data}_ocr')
            else:
                ocrs = []
                for i in dataset:
                    ocrs.extend(i['caption'])
                from datasets import Dataset
                ocr_dataset = Dataset.from_dict({'ocr': ocrs})
                ocr_dataset = ocr_dataset.map(lambda x: {'img': create_text_image(x['ocr'])}, num_proc=40)
                ocr_dataset.save_to_disk(f'{data}_ocr')
        orc_prompt = img_prompt#.replace(' above image ', ' sentence in above image ')
        print(orc_prompt)
        text_embs = emb_data(model,transform, ocr_dataset, device, emb_type='image', prompt=orc_prompt, bsz=bsz)
    else:
        text_embs = emb_data(model,transform, dataset, device, emb_type='text', prompt=text_prompt, bsz=bsz)
    img_embs = emb_data(model,transform, dataset, device, emb_type='image', prompt=img_prompt, bsz=bsz)

    texts_image_index = [i // 5 for i in range(img_embs.shape[0]*5)]
    assert len(texts_image_index) == len(text_embs)

    assert text_embs.isnan().sum().item() == 0, 'nan in retrieve emb'
    assert img_embs.isnan().sum().item() == 0, 'nan in images emb'

    # get the score for each text and image pair
    scores  = text_embs @ img_embs.t()

    positive_pairs = torch.zeros_like(scores, dtype=bool)
    positive_pairs[torch.arange(len(scores)), texts_image_index] = True
    metrics = {}
    recall_k_list = [1, 5, 10]
    batch_size = 64
    for recall_k in recall_k_list:
        # Note that recall_at_k computes **actual** recall i.e. nb_true_positive/nb_positives, where the number
        # of true positives, e.g. for text retrieval, is, for each image,  the number of retrieved texts matching that image among the top-k.
        # Also, the number of positives are the total number of texts matching the image in the dataset, as we have a set of captions
        # for each image, that number will be greater than 1 for text retrieval.
        # However, image/text retrieval recall@k, the way it is done in CLIP-like papers, is a bit different.
        # recall@k, in CLIP-like papers, is, for each image, either 1 or 0. It is 1 if atleast one text matches the image among the top-k.
        # so we can easily compute that using the actual recall, by checking whether there is at least one true positive,
        # which would be the case if the recall is greater than 0. One we compute the recal for each image (or text), we average
        # it over the dataset.
        metrics[f"image_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores, positive_pairs, batch_size, device, k=recall_k)>0).float().mean().item()
        metrics[f"text_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores.T, positive_pairs.T, batch_size, device, k=recall_k)>0).float().mean().item()

    return metrics

def cir(model, transform,
        img_prompt, text_img_prompt,
        data, fiq_data_type,
        device,
        fiq_two=False,
        fusion_cir=False,
        img_only=False,
        batch_size=None):
    print(img_prompt)
    print(text_img_prompt)
    phi3 = MODEL_TYPE == 'phi3'

    if data == 'fashioniq':
        assert fiq_data_type in ['dress', 'shirt', 'toptee']
        dataset = load_dataset('royokong/fashioniq_val')
        img_dataset = load_dataset('royokong/fashioniq_val_imgs')

        dataset = dataset['val'].filter(lambda x: x['category'] == fiq_data_type, num_proc=4)
        img_dataset = img_dataset['val'].filter(lambda x: x['category'] == fiq_data_type, num_proc=4)
    elif data  == 'cirrtest':
        dataset = load_dataset('royokong/cirr_test')
        img_dataset = load_dataset('royokong/cirr_imgs')

        dataset = dataset['test']
        img_dataset = img_dataset['test']
        # skip error of not having target_id
        dataset = dataset.add_column('target_id', [img_dataset[0]['id'] for i in range(len(dataset))])
    else:
        dataset = load_dataset('royokong/cirr_val')
        img_dataset = load_dataset('royokong/cirr_imgs')

        dataset = dataset['val']
        img_dataset = img_dataset['val']

    if DEBUG:
        dataset = dataset.select(range(50))
        img_dataset = img_dataset.select(range(50))

    assert len(set(dataset['target_id']) - set(img_dataset['id'])) == 0

    bsz  = 4
    if fiq_two:
        bsz //= 2
    if batch_size is not None:
        bsz = batch_size

    # emb img
    def custom_collate_fn(batch):
        collated_batch = {}
        for key in batch[0].keys():
            collated_batch[key] = [b[key] for b in batch]
        return collated_batch
    collate_fn = custom_collate_fn

    if phi3: bsz=1

    img_dataloader = torch.utils.data.DataLoader(
        img_dataset, batch_size=bsz,
        shuffle=False, num_workers=4,
        collate_fn=collate_fn
    )
    img_dataloader = accelerator.prepare(img_dataloader)
    images_embs = []
    bar = tqdm(total=len(img_dataloader))
    for batch in img_dataloader:
        input_texts = [img_prompt]*len(batch['img'])
        if phi3:
            assert len(input_texts) == 1
            input_texts = input_texts[0]
        inputs = transform(input_texts,
                           batch['img'], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            embs = model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
            embs = F.normalize(embs, dim=-1)
            assert embs.isnan().sum() == 0, 'nan in emb after norm'
        embs = accelerator.gather(embs)
        images_embs.append(embs.cpu().float())
        bar.update(1)
    images_emb = torch.cat(images_embs)[:len(img_dataset['id'])]
    images_ids = img_dataset['id']

    bar.close()

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=bsz,
        shuffle=False, num_workers=4,
        collate_fn=collate_fn
    )

    retrieve_emb = []
    dataloader = accelerator.prepare(dataloader)
    bar = tqdm(total=len(dataloader))
    for batch in dataloader:
        images = batch['candidate']
        if data == 'fashioniq':
            caption = batch['caption']
            if fiq_two:
                caption = caption + [i[::-1] for i in caption]
                images = images + images
            input_texts = [text_img_prompt.replace('<sent>', ', '.join([cc.strip('.?, ') for cc in c])) for c in caption]
        else:
            input_texts = [text_img_prompt.replace('<sent>', c) for c in batch['caption']]

        if phi3:
            with torch.no_grad():
                _embs = []
                for i in range(len(input_texts)):
                    inputs = transform(input_texts[i],
                                       [images[i],], return_tensors="pt", padding=True).to(device)
                    _embs.append(model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :])
                embs = torch.cat(_embs, dim=0)
                if fiq_two:
                    embs = embs[:len(batch['caption'])] + embs[len(batch['caption']):]
                embs = F.normalize(embs, dim=-1)
        else:
            inputs = transform(input_texts,
                            images, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                embs = model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
                if fiq_two:
                    embs = embs[:len(batch['caption'])] + embs[len(batch['caption']):]
                embs = F.normalize(embs, dim=-1)
        embs = accelerator.gather(embs)
        retrieve_emb.append(embs.cpu().float())
        bar.update(1)
    retrieve_emb = torch.cat(retrieve_emb)[:len(dataset['target_id'])]
    target_ids = dataset['target_id']
    bar.close()

    assert retrieve_emb.isnan().sum().item() == 0, 'nan in retrieve emb'
    assert images_emb.isnan().sum().item() == 0, 'nan in images emb'

    scores  = retrieve_emb @ images_emb.t()


    labels = []
    for i, target_id in enumerate(target_ids):
        labels.append(images_ids.index(target_id))

    if data == 'cirr' or data == 'cirrtest':
    # remove reference itself like SEARLE
        if not DEBUG:
            mask_index = [images_ids.index(label) for label in dataset['candidate_id']]
            for i, mid in enumerate(mask_index):
                scores[i][mid] = -1

    if data == 'cirrtest':
        submission = {
            'version': 'rc2',
            'metric': 'recall'
        }
        pairids = dataset['pairid']
        for i, pairid in enumerate(pairids):
            top_k_indices = torch.topk(scores[i], k=50, largest=True).indices
            submission[str(pairid)] = [images_ids[j] for j in top_k_indices]
        return submission


    def cir_recall_at_k(scores, labels, k):
        """
        Calculate Recall@k using PyTorch
        """
        num_queries = scores.size(0)
        recalls = []
        for i in range(num_queries):
            top_k_indices = torch.topk(scores[i], k=k, largest=True).indices
            recalls.append(int(labels[i] in top_k_indices))
        return sum(recalls) / num_queries

    if data == 'fashioniq':
        # Calculate R@1, R@3, and R@5
        r_at_1 = cir_recall_at_k(scores, labels, 10)
        r_at_5 = cir_recall_at_k(scores, labels, 50)
        metrics = [r_at_1, r_at_5]
    else:
        # Calculate R@1, R@3, and R@5
        r_at_1 = cir_recall_at_k(scores, labels, 1)
        r_at_3 = cir_recall_at_k(scores, labels, 5)
        r_at_5 = cir_recall_at_k(scores, labels, 10)
        metrics = [r_at_1, r_at_3, r_at_5]

    return metrics

def main(
        llava: bool = False,
        llava_llama3: bool = False,
        lora_path: str = None,
        img_only: bool = False,
        eol2: bool = False,
        name: str = None,
        use_icl: bool = False,
        fiq_two: bool = False,
        batch_size: int = 1,
        bf16: bool = False,
        fp32: bool = False,
        use_4bit: bool = False,
        data: str = None,
        not_save_fp32: bool = False,
        e5_project: str = None,
        debug: bool = False,
        ocr_replace_text: bool = False,
        phi3: bool = False,
        use_e5v: bool = False,
):
    global DEBUG, MODEL_TYPE
    DEBUG = debug

    if phi3:
        MODEL_TYPE = 'phi3'
    elif llava_llama3:
        MODEL_TYPE = 'llava_llama3'
    elif use_e5v:
        llava_llama3 = True
        MODEL_TYPE = 'llava_llama3'


    assert MODEL_TYPE in ['llava', 'llava_llama3', 'phi3']

    # set NCCL_DEBUG
    if os.environ.get("NCCL_DEBUG", None) is None:
        os.environ["NCCL_DEBUG"] = "ERROR"

    device=accelerator.device

    model, transform = init_model_and_transform(lora_path, bf16, fp32, use_e5v=use_e5v)
    model.to(device)

    from datasets import disable_caching
    disable_caching()

    datasets = ['flickr30k', 'coco', 'fashioniq dress', 'fashioniq shirt', 'fashioniq toptee', 'cirr']
    if data:
        datasets = data.split(',')

    if ocr_replace_text:
        datasets = ['flickr30k', 'coco']

    all_results = []
    for data in datasets:
        if 'fashioniq' in data:
            data, fiq_data_type = data.split(' ')
            fiq_two = True
        else:
            fiq_data_type = None
            fiq_two = False

        if data == 'flickr30k' or data == 'coco':
            if phi3:
                img_prompt = '<|user|>\n<|image_1|>\nSummary above image in one word:<|end|>\n<|assistant|>\n'
                text_prompt = '<|user|>\n<sent>\nSummary above sentence in one word:<|end|>\n<|assistant|>\n'
            elif llava_llama3:
                img_prompt = llama3_template.format('<image>\nSummary above image in one word: ')
                text_prompt = llama3_template.format('<sent>\nSummary above sentence in one word: ')
            else:
                img_prompt = "[INST] <image>\nSummary above image in one word: [/INST]"
                text_prompt = "[INST] <sent>\nSummary above sentence in one word: [/INST]"

            if accelerator.is_main_process:
                print(img_prompt)
                print(text_prompt)

            metrics = ir(model, transform, img_prompt, text_prompt,
                         data, device, ocr_replace_text, batch_size)
        elif data == 'fashioniq' or data == 'cirr' or data == 'cirrtest':
            if data == 'fashioniq':
                fiq_data_name = fiq_data_type
                if fiq_data_type == 'toptee':
                    fiq_data_name = 'shirt'
                img_prompt = f"[INST] <image>\n Describe this {fiq_data_name} in one word based on its style: [/INST]"
                text_img_prompt = f"[INST] <image> change the style of this {fiq_data_name} to <sent>\n Desribe this modified {fiq_data_name} in one word based on its style: [/INST]"
            else:
                img_prompt = "[INST] <image>\n Describe this image in one word: [/INST]"
                text_img_prompt = "[INST] <image>Modify this image with \"<sent>\", desribe modified image in one word: [/INST]"

            if llava_llama3:
                img_prompt = img_prompt.replace('[INST] ', '').replace(' [/INST]', '')
                text_img_prompt = text_img_prompt.replace('[INST] ', '').replace(' [/INST]', '')
                img_prompt = llama3_template.format(img_prompt)
                text_img_prompt = llama3_template.format(text_img_prompt)

            if phi3:
                img_prompt = img_prompt.replace('[INST] ', '').replace(' [/INST]', '').replace('<image>', '<|image_1|>')
                text_img_prompt = text_img_prompt.replace('[INST] ', '').replace(' [/INST]', '').replace('<image>', '<|image_1|>')

                img_prompt = '<|user|>\n{} <|end|>\n<|assistant|>\n'.format(img_prompt)
                text_img_prompt = '<|user|>\n{} <|end|>\n<|assistant|>\n'.format(text_img_prompt)

            if accelerator.is_main_process:
                print(img_prompt)
                print(text_img_prompt)

            metrics = cir(model, transform, img_prompt, text_img_prompt, data, fiq_data_type,
                          device,
                          fiq_two=fiq_two,
                          batch_size=batch_size)

        if accelerator.is_main_process:
            print(metrics)
            if lora_path is not None or name is not None:
                checkpoint_name = lora_path.replace('/', '_') + '.txt' if lora_path is not None else name
            elif use_e5v:
                checkpoint_name = 'e5v.txt'
            else:
                checkpoint_name = None
            if data == 'cirrtest':
                with open(checkpoint_name.replace('.txt', '') + 'cirr_sub.json', 'w') as f:
                    json.dump(metrics, f)
            else:
                all_results.append(log_to_file(data, metrics, checkpoint_name, fiq_data_type=fiq_data_type, orc_replace_text=ocr_replace_text))

    if accelerator.is_main_process:
        print('\n'.join(all_results))


if __name__ == '__main__':
    from fire import Fire
    Fire(main)
