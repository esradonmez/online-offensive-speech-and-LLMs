import os
import argparse
import pathlib
import json
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score

PROJECT_DIR = pathlib.Path(__file__).resolve().parent

SPLIT_FILES = {
    'hatespeech': 'SBIC.v2/sbic_v2_test.json',
    'microaggressions': 'SBIC.v2/sbic_v2_dev_microaggressions.json',
}

PROMPT_TEMPLATES = ['base_prompt', 'base_prompt_with_exp', 'base_prompt_because']

HF_MODELS = {
    'dolly-v2-3b': 'databricks/dolly-v2-3b',
    'dolly-v2-7b': 'databricks/dolly-v2-7b',
    'dolly-v2-12b': 'databricks/dolly-v2-12b',
    'opt-iml-1.3b': 'facebook/opt-iml-1.3b',
    'opt-iml-30b': 'facebook/opt-iml-30b',
    'falcon-7b': 'tiiuae/falcon-7b-instruct',
    'falcon-40b': 'tiiuae/falcon-40b-instruct',
    'vicuna-7b': 'lmsys/vicuna-7b-v1.5',
    'vicuna-13b': 'lmsys/vicuna-13b-v1.5',
    'vicuna-33b': 'lmsys/vicuna-33b-v1.3',
    'Llama-2-7b': 'meta-llama/Llama-2-7b-chat-hf',
    'Llama-2-13b': 'meta-llama/Llama-2-13b-chat-hf',
    'Llama-2-70b': 'meta-llama/Llama-2-70b-chat-hf',
    'Mistral-7b': 'mistralai/Mistral-7B-Instruct-v0.2',
}

GPT_MODELS = ['gpt-3.5-turbo', 'gpt-4']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, required=True, choices=list(HF_MODELS) + GPT_MODELS, dest='model_name')
    parser.add_argument('--data-dir', type=pathlib.Path, required=True, dest='data_dir')
    parser.add_argument('--model-cache-dir', type=pathlib.Path, dest='model_cache_dir')
    parser.add_argument('--prompt-template', type=str, nargs='+', default=PROMPT_TEMPLATES, dest='prompt_templates')
    parser.add_argument('--split', type=str, nargs='+', default=list(SPLIT_FILES), choices=list(SPLIT_FILES), dest='splits')
    parser.add_argument('--project-dir', type=pathlib.Path, default=PROJECT_DIR, dest='project_dir')
    parser.add_argument('--out', type=str, default='responses', dest='out')

    cfg = parser.parse_args()

    if cfg.model_name in GPT_MODELS:
        template_dir = os.path.join(cfg.project_dir, 'prompt_templates', 'gpt_prompts')
        generate = prompt_gpt
    else:
        if cfg.model_cache_dir is None:
            parser.error('--model-cache-dir is required for Hugging Face models')
        template_dir = os.path.join(cfg.project_dir, 'prompt_templates')
        generate = prompt_hf(cfg)

    for split in cfg.splits:
        for prompt_template in cfg.prompt_templates:
            with open(cfg.data_dir / SPLIT_FILES[split], 'r') as f:
                data = remove_maybe_posts(json.load(f))

            with open(os.path.join(template_dir, prompt_template + '.json'), 'r') as f:
                template_texts = json.load(f)

            print(template_texts)

            output_dir = os.path.join(cfg.project_dir, cfg.out, cfg.model_name, split)
            os.makedirs(output_dir, exist_ok=True)

            generated_answers = generate(cfg, data, template_texts, os.path.join(output_dir, prompt_template + '_temp.json'))

            with open(os.path.join(output_dir, prompt_template + '.json'), 'w') as f:
                json.dump(generated_answers, f, indent=4)

            print(generated_answers['Accuracy'])


def remove_maybe_posts(data):
    new_set = []
    for d in data:
        if d['offensiveYN'] in [0.0, 1.0]:
            new_set.append(d)

    return new_set


def short_answer_to_pred(short_answer):
    if short_answer == 'Yes':
        return 1.0
    elif short_answer == 'No':
        return 0.0
    else:
        return 5.0


def apply_template(template_texts: dict,
                   example: str
                   ):

    text = template_texts['prompt_template']

    prompt_texts = template_texts['prompt_texts']

    text = text.replace('<example>', 'Post: ' + example)
    for key in ['system_prompt', 'prompt_instruction1', 'prompt_instruction2']:
        text = text.replace(f'<{key}>', prompt_texts[key] if prompt_texts[key] else '')

    return text


def load_model(cfg):
    import torch
    from transformers import AutoTokenizer, LlamaTokenizer
    from transformers import LlamaForCausalLM, AutoModelForCausalLM, OPTForCausalLM

    # Gated models (Llama-2, Mistral) are authenticated via the HF_TOKEN environment variable
    model_card = HF_MODELS[cfg.model_name]

    n_gpus = torch.cuda.device_count()
    max_memory = f'{int(49140*n_gpus)}MB'

    tokenizer_class, model_class = AutoTokenizer, AutoModelForCausalLM
    tokenizer_kwargs = {'offload_folder': cfg.model_cache_dir}
    model_kwargs = {}

    if 'falcon' in cfg.model_name:
        tokenizer_kwargs.update(trust_remote_code=True, pad_token_id=11)
        model_kwargs.update(trust_remote_code=True)

    elif 'Mistral' in cfg.model_name:
        model_kwargs.update(pad_token_id=2)

    elif 'vicuna' in cfg.model_name or 'dolly' in cfg.model_name:
        tokenizer_kwargs.update(trust_remote_code=True)
        model_kwargs.update(trust_remote_code=True)

    elif 'Llama' in cfg.model_name:
        tokenizer_class, model_class = LlamaTokenizer, LlamaForCausalLM
        tokenizer_kwargs = {}

    elif 'opt' in cfg.model_name:
        model_class = OPTForCausalLM
        tokenizer_kwargs.update(trust_remote_code=True)

    tokenizer = tokenizer_class.from_pretrained(
        model_card,
        cache_dir=cfg.model_cache_dir,
        **tokenizer_kwargs
    )

    model = model_class.from_pretrained(
        model_card,
        cache_dir=cfg.model_cache_dir,
        device_map="auto", # dispatch efficiently the model on the available ressources
        max_memory = {i: max_memory for i in range(n_gpus)},
        offload_folder=cfg.model_cache_dir,
        **model_kwargs
    )

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return model, tokenizer, device


def prompt_hf(cfg):
    import torch

    model, tokenizer, device = load_model(cfg)

    def generate(cfg, data, template_texts, temp_file):
        labels, preds = [], []

        generated_answers = {
            'Example prompt': '',
            'Results': []
        }

        with torch.no_grad():
            for d in tqdm(data):
                prompt = apply_template(template_texts, d['post'])
                generated_answers['Example prompt'] = prompt

                labels.append(d['offensiveYN'])

                inputs = tokenizer(prompt, return_tensors="pt").to(device)

                # Generate
                generate_ids = model.generate(
                    input_ids = inputs['input_ids'],
                    attention_mask = inputs['attention_mask'],
                    do_sample = False,
                    max_new_tokens = 50,
                    temperature = 0
                )

                output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]

                # trim the generated, keep only the system answer
                generated_answer = output[len(prompt):]

                short_answer = generated_answer.replace('\n', ' ')
                if short_answer != '':
                    if short_answer[0] == ' ':
                        short_answer = short_answer[1:]
                    short_answer = short_answer.split(' ')[0]
                    short_answer = short_answer.replace(',', '')
                    short_answer = short_answer.replace('.', '')

                preds.append(short_answer_to_pred(short_answer))
                d['model answer'] = generated_answer
                d['pred'] = short_answer
                generated_answers['Results'].append(d)

        generated_answers['Accuracy'] = accuracy_score(labels, preds)

        return generated_answers

    return generate


def apply_gpt_prompt(template_texts, example):
    prompt_texts = template_texts['prompt_texts']

    if prompt_texts['system_prompt']:
        messages=[
            {"role": "system", "content": f"{prompt_texts['system_prompt']}"},
            {"role": "user", "content": f"Post: {example}\n{prompt_texts['prompt_instruction1']}"},
            {"role": "user", "content": f"{prompt_texts['prompt_instruction2']}"},
        ]
    else:
        messages=[
            {"role": "user", "content": f"{prompt_texts['prompt_instruction1']}Post: {example}"},
            {"role": "user", "content": f"{prompt_texts['prompt_instruction2']}"},
        ]

    return messages


def prompt_gpt(cfg, data, template_texts, temp_file):
    import openai
    openai.api_key = os.environ["OPENAI_API_KEY"]

    labels, preds = [], []
    generated_answers = {
        'Example prompt': '',
        'Results': []
    }

    for d in tqdm(data):
        messages = apply_gpt_prompt(template_texts, d['post'])
        generated_answers['Example prompt'] = messages

        labels.append(d['offensiveYN'])

        retries = 10
        while retries > 0:
            response = None
            try:
                response = openai.ChatCompletion.create(
                    model=cfg.model_name,
                    messages=messages,
                    request_timeout=15,
                    temperature=0.0,
                    max_tokens=50,
                )
            except Exception as e:
                print(e)
                print('Timeout error, retrying...')
                retries -= 1
                time.sleep(60)
            if response:
                retries = -1
            else:
                with open(temp_file, 'w') as f:
                    json.dump(generated_answers, f, indent=4)

        generated_answer = response["choices"][0]["message"]['content']

        if generated_answer[0] == ' ':
            generated_answer = generated_answer[1:]
        short_answer = generated_answer.split(' ')[0]
        short_answer = short_answer.replace('\n', '')
        short_answer = short_answer.replace(',', '')
        short_answer = short_answer.replace('.', '')

        preds.append(short_answer_to_pred(short_answer))
        d['model answer'] = generated_answer
        d['pred'] = short_answer
        generated_answers['Results'].append(d)

        time.sleep(10)

    generated_answers['Accuracy'] = accuracy_score(labels, preds)

    return generated_answers


if __name__ == '__main__':
    main()
