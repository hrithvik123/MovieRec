from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import inaugural
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

corpus = inaugural.raw('1789-Washington.txt')
print(corpus)

sents = nltk.sent_tokenize(corpus)
print("The number of sentences is", len(sents))

lemmatizer = WordNetLemmatizer()


def lemmatizeSentence(corpus):
    words = word_tokenize(corpus)
    print("The number of tokens is", len(words))
    average_tokens = round(len(words)/len(sents))
    print("The average number of tokens per sentence is", average_tokens)
    unique_tokens = set(words)
    print("The number of unique tokens are", len(unique_tokens))
    stop_words = set(stopwords.words('english'))
    final_tokens = []
    for each in words:
        if each not in stop_words:
            final_tokens.append(each)
    print("The number of total tokens after removing stopwords are",
          len((final_tokens)))
    lemma_sentence = []
    for word in words:
        lemma_sentence.append(lemmatizer.lemmatize(word))
        lemma_sentence.append(" ")
    return "".join(lemma_sentence)


lemma_sentence = print("The lemmatized sentence is:",
                       lemmatizeSentence(corpus))
