## Extract attributes from free-form text
# NER by flair
# keyword by KeyBERT

flair_model_name = 'ner-ontonotes-large'
keybert_model_name = 'all-mpnet-base-v2'

from flair.data import Sentence
from flair.models import SequenceTagger
# load the NER tagger
tagger18 = SequenceTagger.load(flair_model_name)
def flairNER(tagger, text):
    sentence = Sentence(text)
    tagger.predict(sentence)
    # print(sentence)
    # iterate over entities and print
    entities = []
    for entity in sentence.get_spans('ner'):
        entities.append(entity)
    return entities
extract_entities = lambda x: flairNER(tagger18, x)

from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer(keybert_model_name)
from keybert import KeyBERT
kw_model = KeyBERT(model=embedding_model)
extract_keywords = lambda x: kw_model.extract_keywords(x, top_n=20)

def extract_attributes(text):
    entities = extract_entities(text)
    keywords_raw = extract_keywords(text)
    print(entities[0])
    print(type(entities))
    print(keywords_raw[0])
    print(type(keywords_raw))
    print(keywords_raw)

ex1 = """Had dinner at Papa J's with a group of 6.  I loved how the restaurant is in a old brick building with large windows. It felt like a neighborhood restaurant. On a Saturday night, the restaurant was full but not crowded.  We were seated in a room with poor acoustics.  It was difficult to hear people at our table and the waitress.  While she tried, I can see the asperation in her face when she had to repeat the specials to both sides of the table.\n\nPeople ordered bourbon on the rocks before dinner which seemed watered down, while my lemon drop was made nice.  The bread was delicious!  Can you describe it to be creamy?  The fried zucchini was lightly breaded and not too oily.  It was a large portion made up of 2 sliced zucchinis.\n\nWe ordered a variety of dishes.  The pasta dish was dry with more pasta than sauce or meat.  Those who ordered the fish special thought it was delicious.  The shrimp dish was enjoyed as well.  I had the chicken marsala which was pretty good.  The marsala sauce wasn't too thick, and the chicken moist.\n\nHard to tell if the deserts were \"homemade.\"  The tiramisu and spumoni were small in portion and meant for one. \n\nOn the whole, I was on the fence with my overall impression of Papa J's.  \"A-ok\" probably is the best way to describe it."""

if __name__=="__main__":
    extract_attributes(ex1)