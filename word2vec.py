# Python program to generate word vectors using Word2Vec

# importing all necessary modules
import gensim
from gensim.models import Word2Vec
import warnings

warnings.filterwarnings(action = 'ignore')

# # Print results
# print("Cosine similarity between 'alice' " +
# 		"and 'wonderland' - Skip Gram : ",
# 	model2.wv.similarity('alice', 'wonderland'))
	
# print("Cosine similarity between 'alice' " +
# 			"and 'machines' - Skip Gram : ",
# 	model2.wv.similarity('alice', 'machines'))

def getWord2Vec(sentences):
	print(sentences)
	model = Word2Vec(sentences, min_count=1)
	# print(model.wv.most_similar("A", topn=10))
	return model