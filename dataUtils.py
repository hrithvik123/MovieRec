import pandas as pd
import bert
import nlp

dataset = pd.read_csv("./Datasets/FilmTVDataset/filmtv_movies - ENG.csv", nrows= 10000)
print('len before filtering: ' +str(len(dataset)))
dataset = dataset[dataset['country'].str.contains("United States")]
print('len after filtering: ' +str(len(dataset)))
print(dataset.columns)
print("\n")
# print(dataset[dataset["filmtv_id"] == 18]["title"].values[0])


# we only consider title, year, genre, country, directors, actors, description columns
def combine_row(row):
    # corpus = str(row['title']) + ' ' + str(row['year']) + ' ' + str(row['genre']) + ' ' + str(row['country']) + ' ' + str(row['directors']) + ' ' + str(row['actors']) + ' ' + str(row['description'])
    corpus = str(row['genre']) + ' ' + str(row['description'])
    lemmatized_corpus = nlp.lemmatizeSentence(corpus)
    return lemmatized_corpus

dataset['combined_row'] = dataset.apply(combine_row, axis=1)

# bert.getRecommentdation(dataset)