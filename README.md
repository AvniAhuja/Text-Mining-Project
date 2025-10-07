# Text Mining Project

This repository demonstrates a simple, reproducible text mining pipeline:
- Data loading (20 Newsgroups or CSV)
- Preprocessing (tokenization, stopword removal, lemmatization)
- Feature extraction (TF-IDF / CountVectorizer)
- Topic modeling with Gensim (LDA)
- Simple classification with scikit-learn
- Named-entity recognition and sentiment analysis (spaCy + NLTK VADER)

## Quick start

1. Clone and setup environment:
```bash
git clone https://github.com/AvniAhuja/text-mining-project.git
cd text-mining-project
python -m venv venv
source venv/bin/activate     # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

2. Run example:
```bash
python run_example.py
```

3. To use your own CSV:
- Place file in `data/your.csv`
- Use `src.data_loader.load_csv('data/your.csv', text_col='your_text_column', label_col='label')`

## Files
- `src/` — modular code for each pipeline stage
- `notebooks/` — exploratory notebooks (optional)

## License
MIT
