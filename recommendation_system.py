import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import BertModel, BertTokenizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

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
def get_embeddings(text):
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    outputs = bert_model(input_ids)
    embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy()
    return embeddings

# 3. Embedding clustering
def cluster_embeddings(embeddings, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(embeddings)
    return kmeans

# 4. Query extraction using TF-IDF
def extract_query_terms(texts, min_df=0.1, max_df=0.9):
    vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df)
    vectorizer.fit_transform(texts)
    query_terms = vectorizer.get_feature_names()
    return query_terms

# 5. Ranking generation
def generate_ranking(query_terms, actors_data, kmeans):
    query_embeddings = get_embeddings(query_terms)
    cluster_label = kmeans.predict(query_embeddings)[0]
    actors_in_cluster = actors_data[actors_data["cluster"] == cluster_label]

    def score(actor_embeddings, query_embeddings):
        return np.dot(actor_embeddings, query_embeddings.T).sum()

    actors_in_cluster["score"] = actors_in_cluster["embeddings"].apply(
        lambda x: score(x, query_embeddings)
    )
    ranking = actors_in_cluster.sort_values(by="score", ascending=False)
    return ranking

# Example usage
actor_descriptions = [
    "Actor 1 description",
    "Actor 2 description",
    "Actor 3 description",
    # Add more actor descriptions
]

actors_data = pd.DataFrame(
    {
        "actor": [f"Actor {i+1}" for i in range(len(actor_descriptions))],
        "description": actor_descriptions
    }
)

def getRecommendations(dataset, queryDescription):
    dataset["tokens"] = dataset["combined_row"].apply(preprocess)
    dataset["embeddings"] = dataset["tokens"].apply(get_embeddings)
    dataset["embeddings"] = dataset["embeddings"].apply(lambda x: x.reshape(-1))

    all_embeddings = np.vstack(dataset["embeddings"].values)
    kmeans = cluster_embeddings(all_embeddings)

    input_role_description = "The role requires a strong character with great leadership."
    query_terms = extract_query_terms([queryDescription])

    ranking = generate_ranking(query_terms, dataset["combined_row"].tolist(), kmeans)
    print(ranking)
