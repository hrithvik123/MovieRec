import pandas as pd
import bert
import nlp
import tfidf
import word2vec

# # For FilmTV dataset
# dataset = pd.read_csv("./Datasets/FilmTVDataset/filmtv_movies - ENG.csv", nrows= 100)
# dataset.rename(columns={"filmtv_id":"id", "genre":"genres"}, inplace=True)
# dataset = dataset[dataset['country'].str.contains("United States")]

# For netflix dataset
dataset = pd.read_csv("./Datasets/NetflixDataset/titles.csv")
print(dataset.columns)
print("\n")
print('len before filtering: ' +str(len(dataset)))

# edit id column to be autoincrement integers
dataset['id']= pd.Series(range(1,dataset.shape[0]+1))

# filter for movies
dataset = dataset[dataset['type'] == "MOVIE"]

# # filter for production countries to include US
# dataset = dataset[dataset['production_countries'].str.contains("US")]
print('len after filtering: ' +str(len(dataset)))
print("\n")
# print(dataset[dataset["filmtv_id"] == 18]["title"].values[0])


# we only consider title, year, genre, country, directors, actors, description columns
def combine_row(row):
    # corpus = str(row['title']) + ' ' + str(row['year']) + ' ' + str(row['genre']) + ' ' + str(row['country']) + ' ' + str(row['directors']) + ' ' + str(row['actors']) + ' ' + str(row['description'])
    corpus = str(row['genres']) + ' ' + str(row['description'])
    lemmatized_corpus = nlp.lemmatizeSentence(corpus)
    return lemmatized_corpus

dataset['combined_row'] = dataset.apply(combine_row, axis=1)

stringFor202StormCenter = "Alicia Hull (Davis) refuses to remove a pro-Communist book from the library where she works. We are in the deep province of America and it doesn't take long for the library to burn down in retaliation, but she holds out."

# bert.getRecommentdation(dataset)
tfidf.extractQuery(dataset, stringFor202StormCenter)
# word2vec.getWord2Vec(dataset['combined_row'].tolist())