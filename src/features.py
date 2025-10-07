from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def build_tfidf(docs, max_features=5000, ngram_range=(1,2)):
    vect = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X = vect.fit_transform(docs)
    return vect, X

def save_vectorizer(vect, path):
    joblib.dump(vect, path)
