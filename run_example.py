from src.data_loader import load_20newsgroups
from src.preprocess import preprocess_text
from src.features import build_tfidf
from src.classification import train_classifier, report_results
from src.topic_modeling import gensim_preprocess_for_lda, train_lda, print_topics
from src.ner_sentiment import extract_entities, sentiment_score

def main():
    df = load_20newsgroups(subset='train')
    df = df.sample(n=1000, random_state=42).reset_index(drop=True)
    print("Loaded:", df.shape)
    df['clean'] = df['text'].apply(lambda x: preprocess_text(x))
    vect, X = build_tfidf(df['clean'].values, max_features=3000)
    clf, X_test, y_test, preds = train_classifier(X, df['target'].values)
    report_results(y_test, preds)
    tokenized = [doc.split() for doc in df['clean'].tolist()]
    dictionary, corpus = gensim_preprocess_for_lda(tokenized)
    lda = train_lda(dictionary, corpus, num_topics=5, passes=5)
    print_topics(lda, dictionary)
    sample_text = df.loc[0, 'text']
    print("Entities:", extract_entities(sample_text)[:10])
    print("Sentiment:", sentiment_score(sample_text))

if __name__ == "__main__":
    main()
