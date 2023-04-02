import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import BertModel, BertTokenizer

# Initialize required modules
nltk.download("punkt")
nltk.download("wordnet")
lemmatizer = WordNetLemmatizer()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

# 1. Preprocessing and word segmentation
def preprocess(text):
    tokens = word_tokenize(text.lower())
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

# 2. Word/actor embedding learning
def get_embeddings(tokens):
    text = " ".join(tokens)
    input_ids = tokenizer(text, return_tensors="pt", padding=True, truncation=True)["input_ids"]
    outputs = bert_model(input_ids)
    embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy()
    return embeddings

# 3. Embedding clustering
def cluster_embeddings(embeddings, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(embeddings)
    return kmeans

# 4. Query extraction using TF-IDF
def extract_query_terms(texts):
    vectorizer = TfidfVectorizer()
    vectorizer.fit_transform(texts)
    query_terms = vectorizer.get_feature_names_out()
    return query_terms

# 5. Ranking generation
def generate_ranking(query_terms, movies_data, kmeans):
    query_embeddings = get_embeddings(query_terms)
    cluster_label = kmeans.predict(query_embeddings)[0]
    movies_in_cluster = movies_data[movies_data["cluster"] == cluster_label]

    def score(movie_embeddings, query_embeddings):
        return np.dot(movie_embeddings, query_embeddings.T).sum()

    movies_in_cluster["score"] = movies_in_cluster["embeddings"].apply(
        lambda x: score(x, query_embeddings)
    )
    ranking = movies_in_cluster.sort_values(by="score", ascending=False)
    return ranking

# Read the CSV file
csv_file = "./Datasets/NetflixDataset/titles.csv"
movies_data = pd.read_csv(csv_file, nrows=10)

# Preprocess the descriptions
movies_data["tokens"] = movies_data["description"].apply(preprocess)

# Get embeddings for descriptions
movies_data["embeddings"] = movies_data["tokens"].apply(get_embeddings)
movies_data["embeddings"] = movies_data["embeddings"].apply(lambda x: x.reshape(-1))

# Cluster embeddings
all_embeddings = np.vstack(movies_data["embeddings"].values)
kmeans = cluster_embeddings(all_embeddings)
movies_data["cluster"] = kmeans.labels_

# Example usage
input_description = "A thrilling adventure with epic battles and unforgettable characters."
query_terms = extract_query_terms([input_description])

ranking = generate_ranking(query_terms, movies_data, kmeans)
print(ranking)