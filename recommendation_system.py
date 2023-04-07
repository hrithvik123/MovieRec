import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import AutoModel, AutoTokenizer

# Initialize required modules
nltk.download("punkt")
nltk.download("wordnet")
lemmatizer = WordNetLemmatizer()
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/distilbert-base-nli-mean-tokens")
bert_model = AutoModel.from_pretrained("sentence-transformers/distilbert-base-nli-mean-tokens")

# 1. Preprocessing and word segmentation
def preprocess(text):
    tokens = word_tokenize(str(text).lower())
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

# 2. Word/actor embedding learning
def get_embeddings(tokens):
    text = " ".join(tokens)
    input_ids = tokenizer(text, return_tensors="pt", padding=True, truncation=True)["input_ids"]
    outputs = bert_model(input_ids)
    embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy()
    return embeddings[:, :128]  # Limit dimensionality to 128

# 3. Embedding clustering
def cluster_embeddings(embeddings, n_clusters=30):
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
def generate_ranking(query_terms, movies_data, kmeans, top_n=10):
    query_embeddings = get_embeddings(query_terms)
    cluster_label = kmeans.predict(query_embeddings)[0]
    movies_in_cluster = movies_data[movies_data["cluster"] == cluster_label]

    def score(movie_embeddings, query_embeddings):
        return np.dot(movie_embeddings, query_embeddings.T).sum()

    movies_in_cluster["score"] = movies_in_cluster["embeddings"].apply(
        lambda x: score(x, query_embeddings)
    )
    ranking = movies_in_cluster.sort_values(by="score", ascending=False)
    return ranking.head(top_n)

# Calculate Mean Average Precision (MAP)
def evaluate(recommendation_system, test_data, k=10):
    total_map = 0.0
    total_queries = 0

    for index, row in test_data.iterrows():
        input_description = row['description']
        ground_truth_title = row['title']

        recommendations = recommendation_system(input_description)
        titles = recommendations['title'].head(k).tolist()

        relevant_count = 0
        average_precision = 0.0

        for i, title in enumerate(titles):
            if title == ground_truth_title:
                relevant_count += 1
                average_precision += relevant_count / (i + 1)

        average_precision /= min(k, 1)
        total_map += average_precision
        total_queries += 1

    mean_average_precision = total_map / total_queries
    return mean_average_precision


# Read the CSV file
csv_file = "./Datasets/NetflixDataset/titles.csv"
movies_data = pd.read_csv(csv_file)

# Preprocessing
movies_data["tokens"] = movies_data["description"].apply(preprocess)
movies_data["embeddings"] = movies_data["tokens"].apply(get_embeddings)
movies_data["embeddings"] = movies_data["embeddings"].apply(lambda x: x.reshape(-1))

# Cluster embeddings
all_embeddings = np.vstack(movies_data["embeddings"].values)
kmeans = cluster_embeddings(all_embeddings)
movies_data["cluster"] = kmeans.labels_

# Example usage
input_description = "The movie depicts the life of a young boy, Vijay (Amitabh Bachchan), whose father gets brutally lynched by a mobster Kancha Cheena. It's a journey of his quest for revenge, which leads him to become a gangster as an adult. Watch out for Amitabh Bachchan in one of the most powerful roles of his career. Will Vijay lose his family in the process of satisfying his vengeance?"
query_terms = extract_query_terms([input_description])

ranking = generate_ranking(query_terms, movies_data, kmeans, top_n=10)
print(ranking[["title", "score"]])

# For demonstration purposes, we create a mock ground truth dataset
ground_truth = ["Taxi Driver", "Raging Bull", "Goodfellas", "The Deer Hunter", "The Godfather"]

# Convert the ground truth titles to their corresponding ids
ground_truth_ids = ranking[ranking["title"].isin(ground_truth)]["id"].tolist()

# Get the top 10 predicted movie ids
predicted_ids = ranking["id"].head(10).tolist()

# Calculate Mean Average Precision (MAP)
map_score = evaluate(predicted_ids, ground_truth_ids)
print("Mean Average Precision (MAP):", map_score)