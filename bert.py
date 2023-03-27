from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def getId(dataset, title):
    return dataset[dataset.title == title]["filmtv_id"].values[0]

def getTitle(dataset, filmtv_id):
    print("title for: ", str(filmtv_id))
    # print(dataset[dataset["filmtv_id"] == 18]["title"].values[0])
    return str(dataset[dataset["filmtv_id"] == int(filmtv_id)]["title"].values[0])

movie_desc = "Diner"
def getRecommentdation(dataset):
    bert = SentenceTransformer("all-MiniLM-L6-v2")
    sentence_embeddings = bert.encode(dataset["combined_row"].tolist())
    similarity = cosine_similarity(sentence_embeddings)
    movie_reccomendation = sorted(list(enumerate(similarity[getId(dataset, movie_desc)])), key = lambda x:x[1], reverse = True)
    # print(movie_reccomendation[1][0], movie_reccomendation[2][0], movie_reccomendation[3][0])
    recommendationNumber = 1
    x = 0
    while x < 5:
        print('attempt for x: ', str(x))
        try:
            print(getTitle(dataset, int(movie_reccomendation[recommendationNumber][0]))+"\n")
            x += 1
        except IndexError:
            print("An exception occurred for: " +str(recommendationNumber))
        recommendationNumber += 1
    # print(getTitle(int(movie_reccomendation[1][0])), getTitle(int(movie_reccomendation[2][0])), getTitle(int(movie_reccomendation[3][0])), sep = "\n")