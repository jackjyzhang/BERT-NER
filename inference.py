import logging
logging.basicConfig(level=logging.DEBUG)
import pprint
pp = pprint.PrettyPrinter(indent=4)

from datasets import load_dataset
yelp_review = load_dataset("yelp_review_full")
ag_news = load_dataset("ag_news")
yelp_text = [yelp_review['train'][i]['text'] for i in range(0, 100, 10)]
ag_text = [ag_news['train'][i]['text'] for i in range(0, 100, 10)]

def post_process_tags(tags):
    output = []
    for tag in tags:
        parts = tag['tag'].split('-') # B-LOC -> ['B','LOC']
        if len(parts) != 2: 
            # [SEP]
            logging.debug(f'Skipping {tag}')
            continue 
        head = parts[0]; tail = parts[1]; wd = tag['word']
        if len(output) == 0:
            output.append({'tag':tail, 'word':[wd]})
            continue
        if head == 'I' and tail == output[-1]['tag']:
            output[-1]['word'].append(wd)
        else:
            output.append({'tag':tail, 'word':[wd]})
    return output

from bert import Ner
model = Ner("out_large/")
def bertNER(text, ignore_o=True):
    output = model.predict(text)
    output = [e for e in output if e['tag'] != 'O'] if ignore_o else output
    print(text)
    # pp.pprint(output)
    out = post_process_tags(output)
    pp.pprint(out)

from flair.data import Sentence
from flair.models import SequenceTagger
# load the NER tagger
tagger = SequenceTagger.load('flair/ner-english-large')
tagger18 = SequenceTagger.load('flair/ner-english-ontonotes-large')
def flairNER(tagger, text):
    sentence = Sentence(text)
    tagger.predict(sentence)
    # print(sentence)
    # iterate over entities and print
    for entity in sentence.get_spans('ner'):
        print(entity)

from transformers import pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
get_summary = lambda x: summarizer(x, max_length=20, min_length=5, do_sample=False)[0]['summary_text']

from keybert import KeyBERT
kw_model = KeyBERT()
extract_keywords = lambda x: kw_model.extract_keywords(x, keyphrase_ngram_range=(0, 2), use_maxsum=True, nr_candidates=20, top_n=5)

print('Yelp + AG News:')
for text in yelp_text + ag_text:
    print('<bertNER>')
    bertNER(text)
    print('<flairNER>')
    flairNER(tagger, text)
    print('<flairNER 18>')
    flairNER(tagger18, text)
    print('<bart summary>')
    print(get_summary(text))
    print('<keyBERT> keywords')
    pp.pprint(extract_keywords(text))
    print('-'*70)
