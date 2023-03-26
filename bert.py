import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

dataset = pd.read_csv("./Datasets/FilmTVDataset/filmtv_movies - ENG.csv")
print(dataset.columns)
print("\n")


# we only consider title, year, genre, country, directors, actors, description columns
def combine_row(row):
    return str(row['title']) + ' ' + str(row['year']) + ' ' + str(row['genre']) + str(row['country']) + ' ' + str(row['directors']) + ' ' + str(row['actors']) + ' ' + str(row['description'])

def getId(title):
    return dataset[dataset.title == title]["filmtv_id"].values[0]

def getTitle(id):
    return dataset[dataset["filmtv_if"] == id]["title"].valuse[0]

dataset['combined_row'] = dataset.apply(combine_row, axis=1)

bert = SentenceTransformer("all-MiniLM-L6-v2")
sentence_embeddings = bert.encode(dataset["combined_row"].tolist())

similarity = cosine_similarity(sentence_embeddings)

movie_desc = "Uncharted"

movie_reccomendation = sorted(list(enumerate(similarity[getId(movie_desc)])), key = lambda x:x[1], reverse = True)

print(getTitle(movie_reccomendation[1][0]), getTitle(movie_reccomendation[2][0]), getTitle(movie_reccomendation[3][0]), sep = "\n")