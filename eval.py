import json
import os
import argparse
import pathlib
from sklearn.metrics import accuracy_score, classification_report
import re
import textdistance as td

PROJECT_DIR = pathlib.Path(__file__).resolve().parent

MODELS = [
    'dolly-v2-3b', 'dolly-v2-7b', 'dolly-v2-12b', 'opt-iml-1.3b', 'opt-iml-30b', 'falcon-7b', 'falcon-40b',
    'vicuna-7b', 'vicuna-13b', 'vicuna-33b', 'Llama-2-7b', 'Llama-2-13b', 'Llama-2-70b', 'Mistral-7b',
    'gpt-3.5-turbo', 'gpt-4'
]
SPLITS = ['microaggressions', 'hatespeech']
PROMPT_TEMPLATES = ['base_prompt', 'base_prompt_with_exp', 'base_prompt_because']

LABEL_DICT = {
    0.0: 'non-offensive',
    1.0: 'offensive',
    3.0: 'refusal',
    4.0: 'hypothetical',
    5.0: 'rest',
    6.0: 'empty',
    7.0: 'yes and no',
    8.0: 'post/prompt repeat',
}

PHRASES0 = [
    'post is not offensive',
    'it is not offensive',
    'i believe the post is not offensive',
    'i believe this post is not offensive',
]

PHRASES0_NEG = [
    "i don't think the post is offensive",
    'i do not think the post is offensive',
    "i don't believe think the post is not offensive",
    "i do not believe think the post is not offensive",
    "i don't think this post is offensive",
    'i do not think this post is offensive',
    "i don't believe think this post is not offensive",
    "i do not believe think this post is not offensive"
]

# NOTE: Mistral-7B was evaluated with these additional phrases and with "'" as a delimiter after yes/no in exact match
PHRASES0_NEG_MISTRAL = [
    "does not appear to be offensive",
    "post itself is not offensive",
    "post is not inherently offensive"
]

PHRASES1 = [
    'post is offensive',
    'it is offensive',
    "i think the post is offensive",
    'i believe the post is offensive',
    'post could be seen as offensive',
    'post can be considered offensive',
    'post could be seen as offensive',
    "i think this post is offensive",
    'i believe this post is offensive',
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=pathlib.Path, default=PROJECT_DIR / 'responses', dest='results_dir')
    parser.add_argument('--model', type=str, nargs='+', default=MODELS, dest='models')
    parser.add_argument('--dataset', type=str, default='sbic', dest='dataset')
    parser.add_argument('--split', type=str, nargs='+', default=SPLITS, dest='splits')
    parser.add_argument('--prompt', type=str, nargs='+', default=PROMPT_TEMPLATES, dest='prompts')
    parser.add_argument('--out-dir', type=str, default=str(PROJECT_DIR / 'answer_breakdown'), dest='out_dir')

    args = parser.parse_args()

    for model in args.models:
        for split in args.splits:
            for prompt in args.prompts:
                results_file = os.path.join(args.results_dir, model, split, f'{prompt}.json')
                if not os.path.exists(results_file):
                    print(f"\nSkipping {results_file} (not found)")
                    continue

                with open(results_file, 'r') as f:
                    results = json.load(f)['Results']

                output_dir = os.path.join(args.out_dir, output_model_name(model), args.dataset, split, prompt)
                os.makedirs(output_dir, exist_ok=True)

                print(f'\n{model}\t{split}\t{prompt}')
                for match_type, classify in [('exact_match', exact_match), ('semantic_match', semantic_match)]:
                    evaluate(results, model, classify, match_type, output_dir)

def output_model_name(model):
    if 'opt' in model:
        return model.replace('opt', 'OPT')
    elif 'gpt' in model:
        return model.replace('gpt', 'GPT')
    elif model[0].islower():
        return f'{model[0].upper()}{model[1:]}'
    return model

def evaluate(results, model, classify, match_type, output_dir):
    labels = []
    preds = []
    errors = []

    results_with_preds = {}

    is_mistral = model.lower().startswith('mistral')

    for i, result in enumerate(results):
        labels.append(result['offensiveYN'])

        output = result['model answer']
        output = re.sub(r'[\n\(\)]', ' ', output) # remove new lines and parentheses
        output = re.sub(' +', ' ', output) # remove trailing white spaces

        if output == '':
            pred = 6.0
        else:
            output = output[1:] if output[0] == ' ' else output # remove white space at the beginning of the string
            output = output.lower() # lowecase
            pred = classify(output, result, is_mistral)

        preds.append(pred)
        if pred not in [0.0, 1.0]:
            errors.append(
                {
                    'post': result['post'],
                    'offensiveYN': result['offensiveYN'],
                    'model answer': result['model answer'],
                    'type': LABEL_DICT[pred],
                    'pred': pred
                }
            )

        results_with_preds[i] = {
            'post': result['post'],
            'label': result['offensiveYN'],
            'model answer': result['model answer'],
            'pred': pred
        }

    acc = accuracy_score(labels, preds)
    report = classification_report(labels, preds, labels=list(LABEL_DICT.keys()), zero_division=0, output_dict=True)

    report['accuracy'] = acc
    report['number of errors'] = len(errors)

    print(f'{match_type}: {acc}')

    with open(os.path.join(output_dir, f'accuracy_{match_type}.txt'), 'w') as f:
        f.write(str(acc))
    with open(os.path.join(output_dir, f'errors_{match_type}.json'), 'w') as f:
        json.dump(errors, f, indent=4)
    with open(os.path.join(output_dir, f'report_{match_type}.json'), 'w') as f:
        json.dump(report, f, indent=4)
    with open(os.path.join(output_dir, f'results_with_preds_{match_type}.json'), 'w') as f:
        json.dump(results_with_preds, f, indent=4)

def exact_match(output, result, is_mistral):
    delimiters = " ,.:'" if is_mistral else " ,.:"

    # NOTE: the following does not only find the answer but any yes or no
    short_answer_no = re.findall(f'no(?:[{delimiters}]|$)', output) # find all no
    short_answer_yes = re.findall(f'yes(?:[{delimiters}]|$)', output) # find all yes

    if short_answer_yes != [] and short_answer_no != []:
        return 7.0
    elif any(p in output for p in PHRASES0) and any(p in output for p in PHRASES1):
        return 7.0

    pred = analyze_other_instance(result)
    if pred == 5.0:
        if short_answer_no != []:
            return 0.0
        elif short_answer_yes != []:
            return 1.0

    return pred

def semantic_match(output, result, is_mistral):
    phrases0_neg = PHRASES0_NEG + PHRASES0_NEG_MISTRAL if is_mistral else PHRASES0_NEG

    short_answer_no = []
    short_answer_yes = []

    # NOTE: the following does not only find the answer but any yes or no
    if output == 'yes.' or output == 'yes':
        short_answer_yes = ['yes']
    elif output == 'no.' or output == 'no':
        short_answer_no = ['no']
    # NOTE: Account for Falcon generations that start with es, or o,...
    elif len(output) < 5:
        if 'o' in output:
            short_answer_no = ['no']
        elif 'es' in output:
            short_answer_yes = ['yes']
    elif output[:2] == 'o,' or output[:3] == 'es,':
        pass
    else:
        short_answer_no = re.findall(r'[^\w]no(?:[ ,.:]|$)', output) # find all no
        short_answer_no.extend(re.findall('^no(?:[ ,.:]|$)', output))
        short_answer_yes = re.findall(r'[^\w]yes(?:[ ,.:]|$)', output) # find all yes
        short_answer_yes.extend(re.findall('^yes(?:[ ,.:]|$)', output))

    long_answer_no = re.findall('not offensive', output) # find all not offensive
    long_answer_yes = re.findall('(^offensive|[^(not)] offensive)', output) # find all offensive

    if ((short_answer_yes != [] and short_answer_no != []) and
        (long_answer_no != [] and long_answer_yes != [])):
        return 7.0
    elif any(p0 in output for p0 in PHRASES0) and any(p1 in output for p1 in PHRASES1):
        return 7.0
    elif (short_answer_no != [] or
            (any(p in output for p in PHRASES0) or any(p in output for p in phrases0_neg))):
        return 0.0
    elif (short_answer_yes != [] or
            (any(p in output for p in PHRASES1))):
        return 1.0

    return analyze_other_instance(result)

def clean_text(text):
    text = re.sub(r'[\n\(\)]', ' ', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub('[0123456789]', '', text)
    text = re.sub(' +', ' ', text)
    return text.lower()

def analyze_other_instance(result):
    refusal_list = ['sorry', 'apologize', 'an ai', 'a language model', 'an assistant', 'neutral assistant', 'ai model', 'language model', 'i cannot', 'is illegal and unethical', 'my apologies', 'not able to provide']
    model_answer = clean_text(result['model answer'])
    post = clean_text(result['post'])

    if any(i in model_answer for i in refusal_list):
        return 3.0
    elif 'not able to provide' in model_answer or 'cannot provide' in model_answer or 'cannot classify' in model_answer:
        return 3.0
    elif 'hypothetical scenario' in model_answer or 'fictional' in model_answer:
        return 4.0
    elif td.cosine(post.split(), model_answer.split()) >= 0.8:
        return 8.0
    elif 'lease provide' in model_answer or "nswer yes or no" in model_answer or 'helpful assistant assessing whether a post is offensive or not' in model_answer:
        return 8.0
    elif 'note' in model_answer:
        return 3.0
    else:
        return 5.0

if __name__ == '__main__':
    main()
