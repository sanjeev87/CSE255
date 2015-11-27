import gzip
from collections import defaultdict
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline  
from sklearn.linear_model import Ridge
import time
import datetime
import urllib
import scipy.optimize
import random
from collections import defaultdict
import nltk
import string
from nltk.stem.porter import *
from sklearn import linear_model
from nltk.corpus import stopwords
import re
import nltk.data
from nltk import wordpunct_tokenize
import lda
import lda.datasets
import warnings
warnings.filterwarnings('ignore')

# return a set of top words of length size 
def getVocab(reviews_u_b,size):
    wordCounts = defaultdict(int)
    punctuation = set(string.punctuation)
    stemmer = PorterStemmer()
    stops = stopwords.words('english')
    for user in reviews_u_b:
        for business in reviews_u_b[user]:
            r = ''.join([c for c in reviews_u_b[user][business].lower() if not c in punctuation])
            for w in r.split():
                if w in stops:
                    continue
    #             w = stemmer.stem(w)
                wordCounts[w] += 1
    counts = [(wordCounts[w], w) for w in wordCounts]
    counts.sort()
    counts.reverse()
    words = [x[1] for x in counts[:size]]
    sorted(words)
    return words
    
    
# helper that return vocab to index dict 
# each word is mapped to index from 0 to len(vocab)
def getVocabToIndex(vocab):
#     sorted(vocab)
    index = 0
    dict = {}
    for word in vocab:
        dict[word] = index
        index += 1
    return dict

def getBusinessToIndex(reviews_b_u):
    index = 0
    dict = {}
    for business in reviews_b_u:
        dict[business] = index
        index += 1
    return dict
    
def getBusineesToVocabVector(reviews_b_u,vocab):
    # return a dict 
    # key is business ID
    # value is a vector of word counts of length equal to vocab
    punctuation = set(string.punctuation)
    stemmer = PorterStemmer()
    stops = stopwords.words('english')
    bussiness_to_vvector = {}
    vocab_index_dict = getVocabToIndex(vocab)
    for business in reviews_b_u:
        bussiness_to_vvector[business] = np.array([0]*len(vocab))
        for user in reviews_b_u[business]:
            r = ''.join([c for c in reviews_b_u[business][user].lower() if not c in punctuation])
            for w in r.split():
                if w in stops or w not in vocab_index_dict:
                    continue
    #             w = stemmer.stem(w)
                bussiness_to_vvector[business][vocab_index_dict[w]] += 1
    return bussiness_to_vvector
            
def getUserToVocabVector(reviews_u_b,vocab):
    # return a dict 
    # key is user ID
    # value is a vector of word counts of length equal to vocab  
    punctuation = set(string.punctuation)
    stemmer = PorterStemmer()
    stops = stopwords.words('english')
    user_to_vvector = {}
    vocab_index_dict = getVocabToIndex(vocab)
    for user in reviews_u_b:
        user_to_vvector[user] = np.array([0]*len(vocab))
        for business in reviews_u_b[user]:
            r = ''.join([c for c in reviews_u_b[user][business].lower() if not c in punctuation])
            for w in r.split():
                if w in stops or w not in vocab_index_dict:
                    continue
    #             w = stemmer.stem(w)
                user_to_vvector[user][vocab_index_dict[w]] += 1
    return user_to_vvector
    
def trainLDA(reviews_b_u,vocab,num_topics,bussiness_to_vvector):
    # build matrix for each business from getBusineesToVocabVector
    # fit lda model 
    # return lda model
    
    business_to_index = getBusinessToIndex(reviews_b_u)
    
    # train vector X
    X = np.zeros((len(reviews_b_u.keys()),len(vocab))).astype(np.intc)

    for business in reviews_b_u:
        index = business_to_index[business]
        X[index] = (bussiness_to_vvector[business]).astype(np.intc)
    

    #define model
    model = lda.LDA(n_topics=num_topics, n_iter=1500, random_state=1)
    X.astype(np.intc)
    print X
    #fit model
    model.fit(X)
    
    return model
    
def printTopWordsPerTopic(n,model):
    # return a dict
    # return top no words in each topic
    topic_word = model.topic_word_
    n_top_words = n
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))
    
    
def getTopicDist(X,model):
    #given some vocab vector , predict top distribution
    t = model.transform(X, max_iter=20, tol=1e-16)
    return t

def getTopTopics(X,model,num_top):
    n_top_words = num_top
    topic_word = model.topic_word_
    num_topics = topic_word.shape[0]
    vocab_size = topic_word.shape[1]
    t = model.transform(X, max_iter=20, tol=1e-16)
    for i in range(t.shape[0]):
        topic_arr = np.array([a for a in range(num_topics)])
        print "shape:",t.shape
        print "i:",i
        print "dist:",t[i]
        print "for index:", i , " the top topic indices:",(topic_arr)[np.argsort(t[i])][:-(num_top+1):-1]
        
def getUserLDAFeatures(model, user_to_vvector):
    #return dict of user to topic distribution
    user_to_topic_dist = {}
    for user in user_to_vvector:
        user_to_topic_dist[user] = getTopicDist(user_to_vvector[user],model)
    return user_to_topic_dist
    
def getBusinessLDAFeatures(model,business_to_vvector):
    #return dict of business to topic distribution
    business_to_topic_dist = {}
    for business in business_to_vvector:
        business_to_topic_dist[business] = getTopicDist(business_to_vvector[business],model)
    return business_to_topic_dist

def buildRegressionFeatures(model,ratings_u_b,
                            user_to_avg_rating,business_to_avg_rating,reviews_u_b,
                            reviews_b_u,global_avg,vocab):
    # retrun a matrix X
    # every row in the matrix is a feature vector for a given user - business combination
    # retrun a Y vector
    # every row in the Y vector is the the rating that we need to predict for the  feature vector in the corresponding 
    # row of the X matrix
    # return a dict that is a mapping of user -> business -> row number in the feature matrix
    mapping = defaultdict(dict)
    num_user_busi = 0
    for user in reviews_u_b:
        for buss in reviews_u_b[user]:
            num_user_busi += 1
            
    topic_word = model.topic_word_
    num_topics = topic_word.shape[0]
    vocab_size = topic_word.shape[1]
    
    num_feats = num_topics + 3
    X = np.zeros((num_user_busi,num_feats)) # len(vocab) features from LDA + user_avg + item_avg + 1 offset
    Y = np.zeros((num_user_busi,1)) # Ratings - labels
    
    user_to_vvector      = getUserToVocabVector(reviews_u_b,vocab)
    bussiness_to_vvector = getBusineesToVocabVector(reviews_b_u,vocab)
    
    index = 0
    for user in reviews_u_b:
        user_topic_dist = getTopicDist(user_to_vvector[user],model)
#         if sum(user_topic_dist[0]) != 1:
#                 print "sum not 1 for user_topic_dist[0]:",user_topic_dist[0]
#         print "shape of user_topic_dist:",user_topic_dist.shape
        for buss in reviews_u_b[user]:
            buss_topic_dist = getTopicDist(bussiness_to_vvector[buss],model)
#             if sum(buss_topic_dist[0]) != 1:
#                 print "sum not 1 for buss_topic_dist[0]:",buss_topic_dist[0]
# #             print "shape of buss_topic_dist:",buss_topic_dist.shape
            lda_feat = np.array([0.0]*num_feats)
            for i in range(num_topics):
#                 print "i:",i
                lda_feat[i] = user_topic_dist[0][i] * buss_topic_dist[0][i]
            if user in user_to_avg_rating:
                lda_feat[num_topics] = user_to_avg_rating[user] 
            else:
                lda_feat[num_topics] = global_avg 
            if buss in business_to_avg_rating:
                lda_feat[num_topics+1] = business_to_avg_rating[buss]
            else:
                lda_feat[num_topics+1] = global_avg
            lda_feat[num_topics + 2] = 1
            X[index] = lda_feat
            Y[index] = ratings_u_b[user][buss]
            if Y[index] == 0:
                print "Y is 0"
            mapping[user][buss] = index
            index += 1
    return [X,Y,mapping,user_to_vvector,bussiness_to_vvector]

def getAvgVVector(xToVVector, num_topics):
    avg_vec = np.array([0]*num_topics).astype(np.intc)
    # count = 0.0
    # for x in xToVVector:
    #     for v in xToVVector[x]:
    #         avg_vec += v
    #         count += 1
    # avg_vec /= count
    # avg_vec.astype(np.intc)
    # print "returning avg_vec:", avg_vec
    return avg_vec

def build_test_reg_features(model,user_to_vvector,bussiness_to_vvector,
            reviews_u_b,ratings_u_b,global_avg):
    
    mapping = defaultdict(dict)
    num_user_busi = 0
    for user in reviews_u_b:
        for buss in reviews_u_b[user]:
            num_user_busi += 1
            
    topic_word = model.topic_word_
    num_topics = topic_word.shape[0]
    vocab_size = topic_word.shape[1]
    
    avg_user_to_vvector = getAvgVVector(user_to_vvector, num_topics)
    avg_bussiness_to_vvector = getAvgVVector(bussiness_to_vvector, num_topics)
    
    num_feats = num_topics + 3
    X = np.zeros((num_user_busi,num_feats)) #  features from LDA + user_avg + item_avg + 1 offset
    Y = np.zeros((num_user_busi,1)) # Ratings - labels
    
    index = 0
    for user in reviews_u_b:
        user_topic_dist = getTopicDist(avg_user_to_vvector,model)
        if user in user_to_vvector:
            user_topic_dist = getTopicDist(user_to_vvector[user],model)
        # else:
            # print "not present in training user:",user
        for buss in reviews_u_b[user]:
            buss_topic_dist = getTopicDist(avg_bussiness_to_vvector,model)
            if buss in bussiness_to_vvector:
                buss_topic_dist = getTopicDist(bussiness_to_vvector[buss],model)
            # else:
                # print "not present in training business:",buss
            lda_feat = np.array([0.0]*num_feats)
            for i in range(num_topics):
#                 print "i:",i
                lda_feat[i] = user_topic_dist[0][i] * buss_topic_dist[0][i]
            if user in user_to_avg_rating:
                lda_feat[num_topics] = user_to_avg_rating[user] 
            else:
                lda_feat[num_topics] = global_avg 
            if buss in business_to_avg_rating:
                lda_feat[num_topics+1] = business_to_avg_rating[buss]
            else:
                lda_feat[num_topics+1] = global_avg
            lda_feat[num_topics + 2] = 1
            X[index] = lda_feat
            Y[index] = ratings_u_b[user][buss]
            if Y[index] == 0:
                print "Y is 0"
            mapping[user][buss] = index
            index += 1
    return [X,Y,mapping]
    

def getGlobalAvg(reviews_u_b, ratings_u_b):
    avg = 0.0
    count = 0
    for user in reviews_u_b:
        for bussiness in reviews_u_b[user]:
            avg += ratings_u_b[user][bussiness]
            count += 1
    return avg * 1.0 / count

def getBaselineMSE(ratings_u_b,pred):
    se = 0.0
    count = 0
    for user in ratings_u_b:
        for business in ratings_u_b[user]:
            se += pow((ratings_u_b[user][business] - pred),2)
            count += 1
    return se * 1.0 / count

def getMSE(X,Y,theta):
    se = 0.0 
    for i in range(X.shape[0]):
        pred = np.dot(X[i],theta)
        se += pow((pred - Y[i]),2)
    return se * 1.0 / X.shape[0]


def printParams(theta,residuals,rank,s):
    print "theta:",theta,"residuals:",residuals,"rank:",rank,"s:",s

VOCAB_SIZE = 1000
# vocab = getVocab(reviews_u_b,VOCAB_SIZE)
# pickle.dump(vocab, open('vocab.p','wb'))
# load the pickles 

print "start loading pickles"
reviews_b_u = pickle.load( open( "reviews_b_u.p", "rb" ) )
reviews_u_b = pickle.load( open( "reviews_u_b.p", "rb" ) )
reviews_b_u_subset = pickle.load( open( "reviews_b_u_subset.p", "rb" ) )
reviews_u_b_subset = pickle.load( open( "reviews_u_b_subset.p", "rb" ) )
vocab = pickle.load( open( "vocab.p", "rb" ) )
user_to_avg_rating = pickle.load( open( "user_to_avg_rating.p", "rb" ) )
business_to_avg_rating = pickle.load( open( "business_to_avg_rating.p", "rb" ) )
ratings_u_b = pickle.load( open( "ratings_u_b.p", "rb" ) )
ratings_b_u = pickle.load( open( "ratings_b_u.p", "rb" ) )
lda_model = pickle.load( open( "lda_model.p", "rb" ) )

reviews_b_u_dense_test = pickle.load( open( "reviews_b_u_dense_test.p", "rb" ) )
reviews_b_u_dense_train = pickle.load( open( "reviews_b_u_dense_train.p", "rb" ) )
reviews_b_u_dense_val = pickle.load( open( "reviews_b_u_dense_val.p", "rb" ) )

reviews_u_b_dense_test = pickle.load( open( "reviews_u_b_dense_test.p", "rb" ) )
reviews_u_b_dense_train = pickle.load( open( "reviews_u_b_dense_train.p", "rb" ) )
reviews_u_b_dense_val = pickle.load( open( "reviews_u_b_dense_val.p", "rb" ) )

reviews_b_u_sparse_test = pickle.load( open( "reviews_b_u_sparse_test.p", "rb" ) )
reviews_b_u_sparse_train = pickle.load( open( "reviews_b_u_sparse_train.p", "rb" ) )
reviews_b_u_sparse_val = pickle.load( open( "reviews_b_u_sparse_val.p", "rb" ) )

reviews_u_b_sparse_test = pickle.load( open( "reviews_u_b_sparse_test.p", "rb" ) )
reviews_u_b_sparse_train = pickle.load( open( "reviews_u_b_sparse_train.p", "rb" ) )
reviews_u_b_sparse_val = pickle.load( open( "reviews_u_b_sparse_val.p", "rb" ) )


reviews_b_u_medium_test = pickle.load( open( "reviews_b_u_medium_test.p", "rb" ) )
reviews_b_u_medium_train = pickle.load( open( "reviews_b_u_medium_train.p", "rb" ) )
reviews_b_u_medium_val = pickle.load( open( "reviews_b_u_medium_val.p", "rb" ) )

reviews_u_b_medium_test = pickle.load( open( "reviews_u_b_medium_test.p", "rb" ) )
reviews_u_b_medium_train = pickle.load( open( "reviews_u_b_medium_train.p", "rb" ) )
reviews_u_b_medium_val = pickle.load( open( "reviews_u_b_medium_val.p", "rb" ) )

# train the lda model , comment this if you have pre trained pickle
# bussiness_to_vvector = getBusineesToVocabVector(reviews_b_u_sparse_train,vocab)
# lda_model = trainLDA(reviews_b_u_sparse_train,vocab,50,bussiness_to_vvector)


global_avg = getGlobalAvg(reviews_u_b_subset, ratings_u_b)
print "baseline mse:",getBaselineMSE(ratings_u_b,global_avg)

#########  sparse set #############################
# print "build sparse train features"
# X_sparse_train,Y_sparse_train,mapping_sparse_train,user_to_vvector,bussiness_to_vvector = buildRegressionFeatures(lda_model,ratings_u_b,
#                             user_to_avg_rating,business_to_avg_rating,reviews_u_b_sparse_train,
#                             reviews_b_u_sparse_train,global_avg,vocab)

# print "build sparse val features"
# # X_sparse_val,Y_sparse_val,mapping_sparse_val = buildRegressionFeatures(lda_model,ratings_u_b,
# #                             user_to_avg_rating,business_to_avg_rating,reviews_u_b_sparse_val,
# #                             reviews_b_u_sparse_val,global_avg,vocab)



# X_sparse_val,Y_sparse_val,mapping_sparse_val = build_test_reg_features(lda_model,user_to_vvector,bussiness_to_vvector,
#             reviews_u_b_sparse_val,ratings_u_b,global_avg)

# print "build sparse test features"
# # X_sparse_test,Y_sparse_test,mapping_sparse_test = buildRegressionFeatures(lda_model,ratings_u_b,
# #                             user_to_avg_rating,business_to_avg_rating,reviews_u_b_sparse_test,
# #                             reviews_b_u_sparse_test,global_avg,vocab)
# X_sparse_test,Y_sparse_test,mapping_sparse_test = build_test_reg_features(lda_model,user_to_vvector,bussiness_to_vvector,
#             reviews_u_b_sparse_test,ratings_u_b,global_avg)


# theta,residuals,rank,s = np.linalg.lstsq(X_sparse_train, Y_sparse_train.flatten())
# printParams(theta,residuals,rank,s)
# print "reg sparse train mse:", residuals * 1.0 / X_sparse_train.shape[0]
# print "reg sparse val mse:",getMSE(X_sparse_val,Y_sparse_val.flatten(),theta)
# print "reg sparse test mse:",getMSE(X_sparse_test,Y_sparse_test.flatten(),theta)





######### ########   medium set #############################
print "build medium train features"
X_medium_train,Y_medium_train,mapping_medium_train,user_to_vvector,bussiness_to_vvector = buildRegressionFeatures(lda_model,ratings_u_b,
                            user_to_avg_rating,business_to_avg_rating,reviews_u_b_medium_train,
                            reviews_b_u_medium_train,global_avg,vocab)

print "build medium val features"
# X_sparse_val,Y_sparse_val,mapping_sparse_val = buildRegressionFeatures(lda_model,ratings_u_b,
#                             user_to_avg_rating,business_to_avg_rating,reviews_u_b_sparse_val,
#                             reviews_b_u_sparse_val,global_avg,vocab)



X_medium_val,Y_medium_val,mapping_medium_val = build_test_reg_features(lda_model,user_to_vvector,bussiness_to_vvector,
            reviews_u_b_medium_val,ratings_u_b,global_avg)

print "build medium test features"
# X_sparse_test,Y_sparse_test,mapping_sparse_test = buildRegressionFeatures(lda_model,ratings_u_b,
#                             user_to_avg_rating,business_to_avg_rating,reviews_u_b_sparse_test,
#                             reviews_b_u_sparse_test,global_avg,vocab)
X_medium_test,Y_medium_test,mapping_medium_test = build_test_reg_features(lda_model,user_to_vvector,bussiness_to_vvector,
            reviews_u_b_medium_test,ratings_u_b,global_avg)


theta,residuals,rank,s = np.linalg.lstsq(X_medium_train, Y_medium_train.flatten())
printParams(theta,residuals,rank,s)
print "reg sparse train mse:", residuals * 1.0 / X_medium_train.shape[0]
print "reg sparse val mse:",getMSE(X_medium_val,Y_medium_val.flatten(),theta)
print "reg sparse test mse:",getMSE(X_medium_test,Y_medium_test.flatten(),theta)




