import pandas as pd
from sentence_transformers import SentenceTransformer

dataset = pd.read_csv("./Datasets/FilmTVDataset/filmtv_movies - ENG.csv")
print(dataset.columns)


# we only consider title, year, genre, country, directors, actors, description columns
def combine_row(row):
    return row['title'] + ' ' + row['year'] + ' ' + row['genre'] + row['country'] + ' ' + row['directors'] + ' ' + row['actors'] + ' ' + row['description']

dataset['combined_row'] = dataset.apply(combine_row, axis=1)

bert = SentenceTransformer("all-MiniLM-L6-v2")
sentence_embeddings = bert.encode(dataset["combined_row"].toList())