# MovieRec
Two movie recommendation systems using text embeddings based approach.

# Deps
To install dependencies:

```bash
pip3 install -r requirements.txt
```

Alternatively, you could run the following commands:

```bash
pip3 install nltk
pip3 install transformers
pip3 install scikit-learn
pip3 install pandas
pip3 install torch torchvision torchaudio
pip install gensim
```

# How to run the recommendation system ?

To run the BERT recommendation system, run the following command -

```bash
python bert_rs.py
```

To run the Word2Vec recommendation system, run the following command - 

```bash
python w2v_rs.py
```

# Switching datasets for training and testing

By default, Both systems use the Netflix dataset. To use the FilmTV dataset for training and testing, do the following:

In the BERT system, comment out line numbers 92 to 96 and uncomment line numbers 98 to 101.

In the Word2Vec system, comment out line numbers 97 to 101 and uncomment line numbers 103 to 106.
