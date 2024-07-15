import re
import sys
import io, os
import torch
import numpy as np
import logging
import tqdm
import fcntl
import time
import argparse
from prettytable import PrettyTable
import transformers
from transformers import LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

# Set PATHs
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)

def lock_and_write_file(file_path, content):
    with open(file_path, 'a') as file:
        while True:
            try:
                # Acquire an exclusive lock (non-blocking)
                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)

                # Perform your write operations here
                file.write(content + '\n')
                file.flush()

            except IOError as e:
                print("File is locked by another process. Can't write.")
                time.sleep(1)
            finally:
                # Release the lock
                fcntl.flock(file, fcntl.LOCK_UN)
                break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_embedding_sentence_template', type=str, default='*sent_0*\nSummary_above_sentence_in_one_word:')
    parser.add_argument("--tokenizer_name", type=str, default='')
    parser.add_argument("--model_name_or_path", type=str,
                        help="Transformers' model name or path")
    parser.add_argument("--mode", type=str,
                        choices=['dev', 'test', 'fasttest'],
                        default='test',
                        help="What evaluation mode to use (dev: fast mode, dev results; test: full mode, test results); fasttest: fast mode, test results")
    parser.add_argument("--task_set", type=str,
                        choices=['sts', 'transfer', 'full', 'na'],
                        default='sts',
                        help="What set of tasks to evaluate on. If not 'na', this will override '--tasks'")
    parser.add_argument('--load_kbit', type=int,
                        choices=[4,8,16],
                        default=16,
                        help="Load model in kbit")

    parser.add_argument('--avg', action='store_true')


    args = parser.parse_args()
    from accelerate import Accelerator
    accelerator = Accelerator()
    device = accelerator.device

    if args.load_kbit == 4:
        from transformers import BitsAndBytesConfig
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            load_in_4bit=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
            ),
            torch_dtype=torch.float16,
            device_map=device,
        )
    elif 'llava' in args.model_name_or_path or 'e5-v' in args.model_name_or_path:
        from transformers import LlavaNextForConditionalGeneration
        model = LlavaNextForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            load_in_8bit=args.load_kbit == 8,
            torch_dtype=torch.float16,
            device_map=device,
        )
        model = model.language_model
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                     device_map=device,
                                                     output_hidden_states=True,
                                                     _attn_implementation='eager',
                                                     trust_remote_code=True,
                                                     load_in_8bit=args.load_kbit == 8,)


    if 'Phi' in args.model_name_or_path or 'phi' in args.model_name_or_path:
        from transformers import AutoProcessor
        transform = AutoProcessor.from_pretrained("microsoft/Phi-3-vision-128k-instruct", trust_remote_code=True)
        tokenizer = transform.tokenizer
        tokenizer.padding = True
    elif 'llama-3' in args.model_name_or_path or 'e5-v' in args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3-8b-Instruct")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding = True
    elif 'llava' in args.model_name_or_path:
        from transformers import LlavaNextProcessor
        transform = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        tokenizer = transform.tokenizer
        tokenizer.padding = True
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    # Set up the tasks
    #args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
    #args.tasks = ['MR']
    if args.task_set == 'sts':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        if args.mode == 'dev':
            args.tasks = ['STSBenchmark-dev']
    elif args.task_set == 'transfer':
        args.tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
    elif args.task_set == 'full':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        args.tasks += ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']

    # Set params for SentEval
    if args.mode == 'dev' or args.mode == 'fasttest':
        # Fast mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5, 'batch_size': 32}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 32,
                                         'tenacity': 3, 'epoch_size': 2}
    elif args.mode == 'test':
        # Full mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10, 'batch_size':16}
        params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                         'tenacity': 5, 'epoch_size': 4}
    else:
        raise NotImplementedError

    import torch.nn.functional as F

    import torch.distributed as dist
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()


    # SentEval prepare and batcher
    def prepare(params, samples):
        return

    params['batch_size'] = 4*world_size
    def batcher(params, batch, max_length=None):
        # Handle rare token encoding issues in the dataset
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode('utf-8') for word in s] for s in batch]

        sentences = [' '.join(s) for s in batch]
        if max_length == 500:
            sentences = [tokenizer.decode(tokenizer.encode(s, add_special_tokens=False)[:max_length]) for s in sentences]
            max_length = 512

        if args.mask_embedding_sentence_template is not None:
            # *cls*_This_sentence_of_"*sent_0*"_means*mask*.*sep+*
            if 'llama-3' in args.model_name_or_path or 'e5-v' in args.model_name_or_path:
                mllm_template = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n'
            elif 'llava' in args.model_name_or_path:
                mllm_template = '[INST] {} [/INST]'
            elif 'Phi' in args.model_name_or_path or 'phi' in args.model_name_or_path:
                mllm_template = '<|user|>\n{}<|end|>\n<|assistant|>\n'

            template = args.mask_embedding_sentence_template
            template = template.replace('_', ' ').replace('*sep+*', '')\
                                                    .replace('*cls*', '')

            for i, s in enumerate(sentences):
                if len(s) > 0 and s[-1] not in '.?"\'': s += '.'
                s = s.replace('"', '\'')
                if len(s) > 0 and '?' == s[-1]: s = s[:-1] + '.'
                sentences[i] = mllm_template.format(' ' + template.replace('*sent 0*', s).strip())
        real_bsz = len(sentences)
        if real_bsz % world_size != 0:
            sentences += [sentences[0]] * (world_size - real_bsz % world_size)

        bsz = len(sentences)
        sub_sentences = sentences[local_rank * bsz // world_size: (local_rank + 1) * bsz // world_size]

        batch = tokenizer.batch_encode_plus(
            sub_sentences,
            return_tensors='pt',
            padding=True,
            max_length=max_length,
            truncation=max_length is not None
        )

        # Move to the correct device
        for k in batch:
            batch[k] = batch[k].to(device) if batch[k] is not None else None

        # Get raw embeddings
        with torch.no_grad():
            hidden_states = model(output_hidden_states=True, return_dict=True, **batch).hidden_states
            if args.avg:
                last_layer = hidden_states[-1]
                attention_mask = batch['attention_mask'].unsqueeze(-1).expand(last_layer.shape)
                outputs = (last_layer * attention_mask).mean(1)
            else:
                outputs = hidden_states[-1][:, -1, :]

            if outputs.dtype == torch.bfloat16:
                # bfloat16 not support for .numpy()
                outputs = outputs.float()

        emb = outputs
        emb = accelerator.gather(emb)[:real_bsz]
        return emb.cpu()



    results = {}


    args.mask_embedding_sentence_template = args.mask_embedding_sentence_template.replace('\\n', '\n')
    print(args.mask_embedding_sentence_template)

    for task in args.tasks:
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result

    # Print evaluation results
    if args.mode == 'test' or args.mode == 'fasttest':
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                    scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
                else:
                    scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        if accelerator.is_main_process:
            print_table(task_names, scores)
        #
        # write results and template to file
        if args.mask_embedding_sentence_template is not None and args.task_set != 'transfer':
            with open('./sts-org-results', 'a') as f:
                bits = f'{args.load_kbit}bit'
                model_name = args.model_name_or_path.split('/')[-1] + f'({bits})'
                f.write(args.mask_embedding_sentence_template + ' ' + model_name + ' ' + ' '.join([str(s) for s in scores]) + '\n')

        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['acc']))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        if accelerator.is_main_process:
            print_table(task_names, scores)

if __name__ == "__main__":
    main()
