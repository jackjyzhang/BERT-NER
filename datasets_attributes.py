import argparse
from contextlib import redirect_stdout
import pprint

from datasets import load_dataset
from extract import extract_attributes

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', type=str, required=True, help='path to output file')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    yelp_review = load_dataset("yelp_review_full")
    ag_news = load_dataset("ag_news")
    yelp_text = [yelp_review['train'][i]['text'] for i in range(0, 100, 5)]
    ag_text = [ag_news['train'][i]['text'] for i in range(0, 100, 5)]
    data = [(yelp_text, 'yelp review'), (ag_text, 'ag news')]

    with open(args.output, 'w') as f:
        pp = pprint.PrettyPrinter(indent=4, stream=f)
        with redirect_stdout(f):
            for texts, name in data:
                print(f'{name}:')
                for text in texts:
                    entities, keywords = extract_attributes(text)
                    print('Text:')
                    print(text)
                    print('Entities:')
                    pp.pprint(entities)
                    print('Keywords:')
                    pp.pprint(keywords)