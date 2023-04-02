from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def getId(dataset, title):
    return dataset[dataset.title == title]["id"].values[0]

def getTitle(dataset, id):
    print("title for: ", str(id))
    return str(dataset[dataset["id"] == int(id)]["title"].values[0])


def getBertEmbeddings(dataset):
    bert = SentenceTransformer("all-MiniLM-L6-v2")
    sentence_embeddings = bert.encode(dataset["combined_row"].tolist())
    getRecommentdation(dataset, sentence_embeddings)
    return sentence_embeddings

movie_desc = "Taxi Driver"

def getRecommentdation(dataset, sentence_embeddings):
    similarity = cosine_similarity(sentence_embeddings)
    print(list(enumerate(similarity[getId(dataset, movie_desc)])))
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
    return sentence_embeddings