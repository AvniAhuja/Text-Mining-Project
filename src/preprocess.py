import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy

nlp = spacy.load("en_core_web_sm", disable=["parser","ner"])
STOP = set(stopwords.words('english'))

def basic_clean(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    return text.strip()

def preprocess_text(text, do_lemmatize=True):
    text = basic_clean(text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP and len(t) > 1]
    if do_lemmatize:
        doc = nlp(" ".join(tokens))
        tokens = [tok.lemma_ for tok in doc if tok.lemma_ != '-PRON-']
    return " ".join(tokens)
