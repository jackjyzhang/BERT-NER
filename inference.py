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

import yake
# default parameters https://github.com/LIAAD/yake/blob/master/yake/cli.py
# default max_ngram_size is 3
max_ngram_size = 2
custom_kw_extractor = yake.KeywordExtractor(n=max_ngram_size)

print('Yelp + AG News:')
for text in yelp_text + ag_text:
    print('<bertNER>')
    bertNER(text)
    print('<flairNER>')
    flairNER(tagger, text)
    print('<flairNER 18>')
    flairNER(tagger18, text)
    print('<bart summary>')
    summary = get_summary(text)
    print(summary)
    print('<keyBERT> keywords')
    pp.pprint(extract_keywords(text))
    print('<yake keywords>')
    pp.pprint(custom_kw_extractor.extract_keywords(text))
    print('<yake keywords from bart summary>')
    pp.pprint(custom_kw_extractor.extract_keywords(summary))
    print('-'*70)

from transformers import LukeTokenizer, LukeForEntityClassification
tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-open-entity")
model = LukeForEntityClassification.from_pretrained("studio-ousia/luke-large-finetuned-open-entity")
print(model.config.id2label)
example1 = 'Wing sauce is like water. Pretty much a lot of butter and some hot sauce (franks red hot maybe).  The whole wings are good size and crispy, but for $1 a wing the sauce could be better. The hot and extra hot are about the same flavor/heat.  The fish sandwich is good and is a large portion, sides are decent.'
target_words1 = ['fish sandwich', 'water', 'butter', 'hot sauce', 'hot', 'crispy']
example2 = "Unfortunately, the frustration of being Dr. Goldberg's patient is a repeat of the experience I've had with so many other doctors in NYC -- good doctor, terrible staff.  It seems that his staff simply never answers the phone.  It usually takes 2 hours of repeated calling to get an answer.  Who has time for that or wants to deal with it?  I have run into this problem with many other doctors and I just don't get it.  You have office workers, you have patients with medical needs, why isn't anyone answering the phone?  It's incomprehensible and not work the aggravation.  It's with regret that I feel that I have to give Dr. Goldberg 2 stars."
target_words2 = ['Dr. Goldberg', 'NYC', '2 hours']

inputs = [(example1, target_words1), (example2, target_words2)]

def get_entity_class(example, target_word):
    entity_spans = [(example.find(target_word), example.find(target_word)+len(target_word))]  # character-based entity span corresponding to "Beyoncé"
    inputs = tokenizer(example, entity_spans=entity_spans, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    print(f"word: {target_word}, predicted class: {model.config.id2label[predicted_class_idx]}")

for example, target_words in inputs:
    for word in target_words:
        get_entity_class(example, word)

