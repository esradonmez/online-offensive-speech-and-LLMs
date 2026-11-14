# online-offensive-speech-and-LLMs

Code for the paper [Please note that I'm just an AI: Analysis of Behavior Patterns of LLMs in (Non-)offensive Speech Identification](https://aclanthology.org/2024.emnlp-main.1019/) (EMNLP 2024).

## Contents

| Path | What it is |
|---|---|
| `prepare_data.py` | Builds the hate speech and microaggressions sets from the SBIC files |
| `prompt_LM.py` | Prompts a model on SBIC hate speech (test split) and microaggressions (dev split) with the three prompt templates |
| `eval.py` | Classifies the model responses (exact and semantic match) and writes the results to `answer_breakdown/` |
| `analysis.ipynb` | Produces the figures and result tables from `answer_breakdown/` |
| `prompt_templates/` | The three prompt templates (T1 `base_prompt`, T2 `base_prompt_with_exp`, T3 `base_prompt_because`); `gpt_prompts/` holds the chat-format versions |
| `data/` | Profanity and stereotype word lists used in the analysis |

## Setup

```
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

Download the [Social Bias Inference Corpus](https://maartensap.com/social-bias-frames/) (`SBIC.v2.tgz`), extract it into a folder named `SBIC.v2`, and prepare the two evaluation sets:

```
python prepare_data.py --sbic-dir <path to SBIC.v2>
```

This writes `sbic_v2_test.json` (hate speech, the test split) and `sbic_v2_dev_microaggressions.json` (microaggressions, the posts from the dev split with source `redditMicroagressions`) into the same folder. Each post gets the most common offensiveness label among its annotators; posts whose label is neither 0 nor 1 are skipped when prompting.

## Usage

```
python prompt_LM.py --model-name vicuna-7b --data-dir <folder containing SBIC.v2> --model-cache-dir <model cache>
python prompt_LM.py --model-name gpt-4 --data-dir <folder containing SBIC.v2>
python eval.py
jupyter notebook analysis.ipynb
```

`prompt_LM.py` runs both splits and all three templates for one model and writes to `responses/<model>/<split>/<template>.json`. Available models: `dolly-v2-3b`, `dolly-v2-7b`, `dolly-v2-12b`, `opt-iml-1.3b`, `opt-iml-30b`, `falcon-7b`, `falcon-40b`, `vicuna-7b`, `vicuna-13b`, `vicuna-33b`, `Llama-2-7b`, `Llama-2-13b`, `Llama-2-70b`, `Mistral-7b`, `gpt-3.5-turbo`, `gpt-4`.

Gated Hugging Face models need a token in `HF_TOKEN`; the GPT models need `OPENAI_API_KEY`.

## Notes

- `Mistral-7b` needs `transformers>=4.40`; the other models were run with the version pinned in `requirements.txt`.
- The responses are not included because of their size. They can be regenerated with `prompt_LM.py`.

## Word lists

- `data/profane_words.json` is the profanity list `words.json` from [zacanger/profane-words](https://github.com/zacanger/profane-words) (version of 20 August 2023, commit `b1843a2`, 2,822 entries).
- `data/stereo_dict.pkl` is the stereotype lexicon `data/stereo_dict.pkl` from [myracheng/markedpersonas](https://github.com/myracheng/markedpersonas), published with [Marked Personas: Using Natural Language Prompts to Measure Stereotypes in Language Models](https://aclanthology.org/2023.acl-long.84/) (Cheng et al., ACL 2023) and based on the word lists of Ghavami and Peplau (2013).

## Citation

```
@inproceedings{donmez-etal-2024-please,
    title = "Please note that {I}{'}m just an {AI}: Analysis of Behavior Patterns of {LLM}s in (Non-)offensive Speech Identification",
    author = {D{\"o}nmez, Esra  and
      Vu, Thang  and
      Falenska, Agnieszka},
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.1019/",
    doi = "10.18653/v1/2024.emnlp-main.1019",
    pages = "18340--18357"
}
```
