import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import AutoModel, AutoTokenizer
from nltk.corpus import stopwords
from string import punctuation
import re

# Utility function to get a list of movie ids matching a given input description. 
# This list would be used as ground truth since we are testing with the same data that is a part of out training dataset
def getGroundTruth(dataset, desc):
    # print(dataset[dataset["description"] == desc]["title"].values)
    return dataset[dataset["description"] == desc]["id"].values

# Initialize required modules
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Load pretrained bert model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/distilbert-base-nli-mean-tokens")
bert_model = AutoModel.from_pretrained("sentence-transformers/distilbert-base-nli-mean-tokens")

# method that checks if a given description contains more than just digits, punctuation and stop words
# description is consider meaningful if true 
def is_meaningful(description):
    words = re.findall(r'\w+', description.lower())
    meaningful_words = [word for word in words if word not in stop_words and word.isalpha()]
    return len(meaningful_words) > 0

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
def evaluate(recommendation_system, test_data):
    total_correct = 0

    for id in recommendation_system:
        if(id in test_data):
            total_correct += 1

    precision = total_correct / len(recommendation_system)
    return precision

test_description = "The movie depicts the life of a young boy, Vijay (Amitabh Bachchan), whose father gets brutally lynched by a mobster Kancha Cheena. It's a journey of his quest for revenge, which leads him to become a gangster as an adult. Watch out for Amitabh Bachchan in one of the most powerful roles of his career. Will Vijay lose his family in the process of satisfying his vengeance?"

# Read and filter the Netflix dataset CSV
csv_file = "./Datasets/NetflixDataset/titles.csv"
movies_data = pd.read_csv(csv_file)
# filter for movies
movies_data = movies_data[movies_data['type'] == "MOVIE"]

# # Read and modify the FilmTV dataset CSV
# csv_file = "./Datasets/FilmTVDataset/filmtv_movies - ENG.csv"
# movies_data = pd.read_csv(csv_file)
# movies_data.rename(columns={"filmtv_id":"id"}, inplace=True)

print("number of rows in csv: ", len(movies_data))

# only keep rows where description is not empty.
movies_data = movies_data[movies_data['description'].notna()]

# only keep rows with meaningful descriptions, i.e.,
# rows with more than just punctuation, digits and stopwords.
movies_data = movies_data[movies_data['description'].apply(is_meaningful)]

# edit id column to be autoincrement integers
movies_data['id']= pd.Series(range(1,movies_data.shape[0]+1))

print("number of movies in csv: ", len(movies_data))

# Preprocessing
movies_data["tokens"] = movies_data["description"].apply(preprocess)
movies_data["embeddings"] = movies_data["tokens"].apply(get_embeddings)
movies_data["embeddings"] = movies_data["embeddings"].apply(lambda x: x.reshape(-1))

# Cluster embeddings
all_embeddings = np.vstack(movies_data["embeddings"].values).astype(np.float32)
kmeans = cluster_embeddings(all_embeddings)
movies_data["cluster"] = kmeans.labels_

def bert_recommendation_system_test(test_data_size = 10):

    # if the size of test data provided is greater than total rows in csv, update it.
    test_data_size = len(movies_data) if len(movies_data) < test_data_size else test_data_size
    
    # For every query, we calculate precision and then we find the average precision for the queries.
    # Declare a variable to keep track of total precision after each query.
    total_precision = 0

    # Example usage
    for description in movies_data.head(test_data_size)["description"].values:
        test_description = description
        query_terms = extract_query_terms([test_description])
        movies_data.loc[movies_data["description"] == test_description, "query_terms"] = ', '.join(query_terms)

        ranking = generate_ranking(query_terms, movies_data, kmeans, top_n=1)
        print(ranking[["title", "score"]])

        ground_truth = getGroundTruth(movies_data, test_description)

        # Convert the ground truth titles to their corresponding ids
        # ground_truth_ids = ground_truth["id"].values

        # Get the top predicted movie ids
        predicted_ids = ranking["id"].values

        # Calculate Mean Average Precision (MAP)
        map_score = evaluate(predicted_ids, ground_truth)
        total_precision += map_score
        movies_data.loc[movies_data["description"] == test_description, "precision"] = map_score

    return total_precision/test_data_size
    

# Method that runs multiple tests on our recommendation system
def rs_multiple_tests():
    test_sizes = [10, 100, 1000, 3000]
    results = {}
    for test_size in test_sizes:
        print("Begin testing with size : ", test_size)
        results[str(test_size)] = bert_recommendation_system_test(test_size)

    print("\nUSING FILE ", csv_file, "FOR INPUT AND TEST DATA:")
    for test_size in test_sizes:
        print("TESTING THE BERT SYSTEM ON FIRST", test_size, " ROWS. MEAN PRECISION =", results[str(test_size)])
    
    # If you want to check the intermediate outputs of all 5 steps and the precision, you can output movies_data to csv
    # Uncomment the line below to export to csv
    # movies_data.to_csv("movies_data_bert.csv")

rs_multiple_tests()