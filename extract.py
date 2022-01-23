## Extract attributes from free-form text
# NER by flair
# keyword by KeyBERT
NOUN = 'NOUN'
CARDINAL = 'CARDINAL'
DATE = 'DATE'
EXCLUDE_NER = [CARDINAL, DATE]
ngram_range = (1, 2)

import os
offline= bool(os.getenv('TRANSFORMERS_OFFLINE') or os.getenv('HF_DATASETS_OFFLINE'))
flair_ner_model = 'ner-ontonotes-large'
keybert_model_name = 'all-mpnet-base-v2' if offline else 'sentence-transformers/all-mpnet-base-v2'
flair_pos_model = 'upos'
summary_model_name = 'facebook/bart-large-cnn'

from flair.data import Sentence
from flair.models import SequenceTagger
# load the NER tagger
tagger18 = SequenceTagger.load(flair_ner_model)
def flairNER(tagger, text):
    sentence = Sentence(text)
    tagger.predict(sentence)
    entities = sentence.to_dict(tag_type='ner')['entities']
    filtered = [e for e in entities if e['labels'][0].value not in EXCLUDE_NER]
    return filtered
extract_entities = lambda x: flairNER(tagger18, x)

from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer(keybert_model_name)
from keybert import KeyBERT
kw_model = KeyBERT(model=embedding_model)
extract_keywords = lambda x: kw_model.extract_keywords(x, keyphrase_ngram_range=ngram_range, top_n=20)
extract_keywords_mmr = lambda x: kw_model.extract_keywords(x, keyphrase_ngram_range=ngram_range, use_mmr=True, diversity=0.1, top_n=20)

import nltk
sno = nltk.stem.SnowballStemmer('english')
stem_word = lambda x: sno.stem(x)

def extract_attributes(text, keywords_lim=5, use_mmr=False):
    entities = extract_entities(text)
    entities_text = [e['text'].lower().split(' ') for e in entities]
    entities_words = set(item for sublist in entities_text for item in sublist)
    entities_stem = [stem_word(word) for word in entities_words]
    entities_ws = entities_words.union(entities_stem)
    keywords_raw = extract_keywords_mmr(text) if use_mmr else extract_keywords(text)
    keywords = [keyword for keyword in keywords_raw if not (keyword[0] in entities_ws or stem_word(keyword[0]) in entities_ws)]
    keywords = []
    keywords_stem = []
    for keyword in keywords_raw:
        kwtext = keyword[0]
        kwstem = stem_word(keyword[0])
        if kwtext in entities_ws or kwstem in entities_ws:
            continue
        if kwstem in keywords_stem:
            continue
        keywords.append(keyword)
        keywords_stem.append(kwstem)
    keywords_trim = keywords[:keywords_lim]
    return entities, keywords_trim

# --- from summary ---

pos_tagger = SequenceTagger.load(flair_pos_model)
def flairPOS(text):
    sentence = Sentence(text)
    pos_tagger.predict(sentence)
    return sentence.to_dict(tag_type='pos')['entities']

from transformers import pipeline
summarizer = pipeline("summarization", model=summary_model_name)
def extract_attributes_summary(text, max_length=20, min_length=5, do_sample=False):
    summary_text = summarizer(text, max_length=max_length, min_length=min_length, do_sample=do_sample)[0]['summary_text']
    pos_list = flairPOS(summary_text)
    nouns = [pos for pos in pos_list if pos['labels'][0].value == NOUN]
    return summary_text, nouns

def extract_atrributes_words(text, do_summary=False, keywords_lim=5, use_mmr=False, max_length=20, min_length=5, do_sample=False):
    entities, keywords = extract_attributes(text, keywords_lim=keywords_lim, use_mmr=use_mmr)
    atts = [e['text'] for e in entities if e['labels'][0].value != CARDINAL] + [e[0] for e in keywords]
    if do_summary:
        summary_text, nouns = extract_attributes_summary(text, max_length=max_length, min_length=min_length, do_sample=do_sample)
        print(f'Summary: {summary_text}')
        atts += [e['text'] for e in nouns]
    return set(atts)


if __name__=="__main__":
    ex = ["""Had dinner at Papa J's with a group of 6.  I loved how the restaurant is in a old brick building with large windows. It felt like a neighborhood restaurant. On a Saturday night, the restaurant was full but not crowded.  We were seated in a room with poor acoustics.  It was difficult to hear people at our table and the waitress.  While she tried, I can see the asperation in her face when she had to repeat the specials to both sides of the table.\n\nPeople ordered bourbon on the rocks before dinner which seemed watered down, while my lemon drop was made nice.  The bread was delicious!  Can you describe it to be creamy?  The fried zucchini was lightly breaded and not too oily.  It was a large portion made up of 2 sliced zucchinis.\n\nWe ordered a variety of dishes.  The pasta dish was dry with more pasta than sauce or meat.  Those who ordered the fish special thought it was delicious.  The shrimp dish was enjoyed as well.  I had the chicken marsala which was pretty good.  The marsala sauce wasn't too thick, and the chicken moist.\n\nHard to tell if the deserts were \"homemade.\"  The tiramisu and spumoni were small in portion and meant for one. \n\nOn the whole, I was on the fence with my overall impression of Papa J's.  \"A-ok\" probably is the best way to describe it.""", # yelp
    """Science, Politics Collide in Election Year (AP) AP - With more than 4,000 scientists, including 48 Nobel Prize winners, having signed a statement opposing the Bush administration's use of scientific advice, this election year is seeing a new development in the uneasy relationship between science and politics.""", # agnews
    """Terrible movie. Nuff Said.<br /><br />These Lines are Just Filler. The movie was bad. Why I have to expand on that I don't know. This is already a waste of my time. I just wanted to warn others. Avoid this movie. The acting sucks and the writing is just moronic. Bad in every way. The only nice thing about the movie are Deniz Akkaya's breasts. Even that was ruined though by a terrible and unneeded rape scene. The movie is a poorly contrived and totally unbelievable piece of garbage.<br /><br />OK now I am just going to rag on IMDb for this stupid rule of 10 lines of text minimum. First I waste my time watching this offal. Then feeling compelled to warn others I create an account with IMDb only to discover that I have to write a friggen essay on the film just to express how bad I think it is. Totally unnecessary.""", # imdb
    'with pale blue berries. in these peaceful shades--', 'it flows so long as falls the rain,', 'and that is why, the lonesome day,', # poetry
    '''Optical mice use an LED and a camera to rapidly capture images of the surface beneath the mouse.  The infomation from the camera is analyzed by a DSP (Digital Signal Processor) and used to detect imperfections in the underlying surface and determine motion. Some materials, such as glass, mirrors or other very shiny, uniform surfaces interfere with the ability of the DSP to accurately analyze the surface beneath the mouse.  \\nSince glass is transparent and very uniform, the mouse is unable to pick up enough imperfections in the underlying surface to determine motion.  Mirrored surfaces are also a problem, since they constantly reflect back the same image, causing the DSP not to recognize motion properly. When the system is unable to see surface changes associated with movement, the mouse will not work properly.''', # yahoo answer
    ] 
    for e in ex:
        # print(extract_attributes(e))
        # print(extract_attributes_summary(e))
        print(e)
        print(extract_atrributes_words(e))
        print(extract_atrributes_words(e, do_summary=True))
        print('---')