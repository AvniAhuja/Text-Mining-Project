from gensim import corpora
from gensim.models import LdaModel

def gensim_preprocess_for_lda(docs_tokens):
    dictionary = corpora.Dictionary(docs_tokens)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in docs_tokens]
    return dictionary, corpus

def train_lda(dictionary, corpus, num_topics=5, passes=5, random_state=42):
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics,
                   passes=passes, random_state=random_state)
    return lda

def print_topics(lda, dictionary, num_words=10):
    for i in range(lda.num_topics):
        print(f"Topic {i}: ", lda.show_topic(i, num_words))
