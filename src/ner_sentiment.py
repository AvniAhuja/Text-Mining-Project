import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nlp = spacy.load("en_core_web_sm")
sia = SentimentIntensityAnalyzer()

def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def sentiment_score(text):
    return sia.polarity_scores(text)
