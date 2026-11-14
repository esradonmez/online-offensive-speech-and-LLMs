import argparse
import json
import pathlib
from collections import Counter

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sbic-dir', type=pathlib.Path, required=True, dest='sbic_dir')

    cfg = parser.parse_args()

    test = read_annotations(cfg.sbic_dir / 'SBIC.v2.tst.csv')
    with open(cfg.sbic_dir / 'sbic_v2_test.json', 'w') as f:
        json.dump(group_by_post(test), f, indent=4)

    dev = read_annotations(cfg.sbic_dir / 'SBIC.v2.dev.csv')
    microaggressions = [item for item in dev if item['dataSource'] == "redditMicroagressions"]
    with open(cfg.sbic_dir / 'sbic_v2_dev_microaggressions.json', 'w') as f:
        json.dump(group_by_post(microaggressions), f, indent=4)


def read_annotations(csv_file):
    df = pd.read_csv(csv_file)
    return json.loads(df.to_json(orient='records'))


def group_by_post(annotations):
    posts = {}
    for item in annotations:
        posts.setdefault(item['post'], []).append(item)

    new_set = []
    for post, annotators in posts.items():
        # the post label is the most common offensiveness label among its annotators
        value, count = Counter(annotator['offensiveYN'] for annotator in annotators).most_common()[0]
        new_set.append(
            {
                'post': post,
                'annotations': annotators,
                'offensiveYN': value
            }
        )

    return new_set


if __name__ == '__main__':
    main()
