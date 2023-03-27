from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results


def extractQuery(dataset, queryString):
    #load a set of stop words
    stopWordsList = stopwords.words('english')
    #get the text column 
    docs=dataset['combined_row'].tolist()
    #create a vocabulary of words, 
    #ignore words that appear in 85% of documents, 
    #eliminate stop words
    cv = CountVectorizer(max_df=0.85,stop_words=stopWordsList)
    word_count_vector = cv.fit_transform(docs)
    print(list(cv.vocabulary_.keys())[:10])
    
    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(word_count_vector)
    
    # you only needs to do this once, this is a mapping of index to 
    feature_names = cv.get_feature_names_out()
    #generate tf-idf for the given document
    tf_idf_vector = tfidf_transformer.transform(cv.transform([queryString]))
    #sort the tf-idf vectors by descending order of scores
    sorted_items = sort_coo(tf_idf_vector.tocoo())
    #extract only the top n; n here is 10
    keywords = extract_topn_from_vector(feature_names,sorted_items,10)
    # now print the results
    print("\n=====Doc=====")
    print(queryString)
    print("\n===Keywords===")
    for k in keywords:
        print(k,keywords[k])
