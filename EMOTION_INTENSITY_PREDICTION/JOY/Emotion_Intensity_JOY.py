#!/usr/bin/env python
# coding: utf-8

# # Emotion Intensity Prediction - JOY

# ## Import libraries

# In[1]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.svm import SVR
from pprint import pprint
from scipy import stats
from numpy import sqrt
from time import time
import pandas, re


# ## Global variables for files, lists & dicts

# In[2]:


## global vars for file
TRAIN_DATA_PATH = 'joy_train.csv'
TEST_DATA_PATH = 'joy_test.csv'
HASHTAG_EMOTION_LEXICON = 'joy_hashtag.csv'
WORD_EMOTION_LEXICON = 'joy_emotion.csv'
EMOTION_EXPANDED = 'emotion_expanded.csv'


print("Started the stopwatch...")
start_time = time()

# global vars
X_Train = []            # training set with features
Y_Train = []            # list of all emotion value results from training set
Train_Sentence = []     # list of all sentences from joy training set

X_Test = []             # testing set with features
Y_Test = []             # list of all emotion value results from testing set
Test_Sentence = []      # list of all sentences from joy testing set

SIA_Vader = SentimentIntensityAnalyzer()

joy_hashtag = {}        # dict with keys as hashtag words & values as joy emotions
word_emotion = {}       # dict with keys as words that represent joy, values = boolean True
emotion_expanded = {}   # dict with keys as words & values with thier joy rate


# ## Load required files
# **Syntax -**
# + File name in lexicon = FILE_NAME_ALIAS : INFO
# <br>
# 1. **'5. NRC-Hashtag-Emotion-Lexicon-v0.2.txt'** = HASHTAG_EMOTION_LEXICON : Contains hashtag words with their respective joy values
# 2. **'8. NRC-word-emotion-lexicon.txt'** = WORD_EMOTION_LEXICON : Contains list of emotion words which represent joy
# 3. **'6. NRC-10-expanded.csv'** = EMOTION_EXPANDED : Contains various emotion intensity for joy words

# In[3]:


#### load files

with open(HASHTAG_EMOTION_LEXICON, 'r') as f:
    for i in f:
        line = list(map(str,i.split()))
        joy_hashtag[line[1]] = float(line[2])

with open(WORD_EMOTION_LEXICON, 'r') as f:
    for i in f:
        line = list(map(str,i.split()))
        word_emotion[line[0]] = True

exp_emo = pandas.read_csv(EMOTION_EXPANDED, sep = "\t")
for i in range(len(exp_emo)):
    emotion_expanded[exp_emo['word'][i]] = float(exp_emo["joy"][i])     # dict = 'word' : float(joy value)

print("Time to load files ->",time() - start_time)


# ## Feature functions
# 1. check_elongation : Counts the freq of elongated words in a tweet
# 2. check_hashtag : Counts the freq of hashtags used in a tweet
# 3. check_CAPS : Counts the freq of CAPS words in a tweet
# 4. check_tag : Counts the freq of tagged people
# 5. check_negation : Counts the freq of negative words in a tweet
# 6. check_word_emotion : Count the freq of words found in **word_emotion** in a tweet
# 7. check_joy_hashtag : Return the average score of words found in **joy_hashtag** in a tweet
# 8. check_exp_emo : Return the average score of words found in **emotion_expanded** in a tweet
# 9. VADER : Append the list of dict values {'pos', 'neu', 'neg', 'compound'} as given by polarity_score
# 
# * More features ahead like punctuations, Count Vectorization (unigram + bigram)

# In[4]:


############### features #################

def check_elongation(word):
    temp = re.sub(r'(.)\1+', r'\1\1', word)
    if len(temp) != len(word):
        return 1
    else:
        return 0

def check_hashtag(word):
    if word[0] != "#":
        return 0
    return 1

def check_CAPS(word):
    if word.isupper():
        return 1
    return 0

def check_tag(word):
    if word[0] != "@":
        return 0
    return 1

def check_negation(word):
    if word.lower() in ['not', 'no', 'nope', 'nopes', 'never', 'neither', 'nor', 'none']:
        return 1
    return 0

def check_word_emotion(word):
    if word.lower() not in word_emotion:
        return 0
    return 1

def check_joy_hashtag(line):
    score = []
    for word in line:
        if word.lower() in joy_hashtag:
            score.append(joy_hashtag[word.lower()])
    if score != []:
        return sum(score)/len(score)
    return 0

def check_exp_emo(line):
    score = []
    for word in line:
        if word.lower() in emotion_expanded:
            score.append(emotion_expanded[word.lower()])
    if score != []:
        return sum(score)/len(score)
    return 0    

def VADER(X_T, sentences):
    count = 0
    for i in sentences:
        score = SIA_Vader.polarity_scores(i)
        X_T[count].append(score['pos'])
        X_T[count].append(score['neu'])
        X_T[count].append(score['neg'])
        X_T[count].append(score['compound'])
        count+=1


# ## Preprocess function
# * Function to make dataset for **X_Train** , **Y_Train** , **X_Test** , **Y_Test**
# * Adding all the features
# * Make list of Training & Testing sentences

# In[5]:


################ preprocess function ########################
 
def TT_preprocess(dataset, sentences, Y_T, X_T):    # preprocess Train & Test data
    for i in dataset:
        X_T.append([0,0,0,0,0,0,0,0,])              # append feaures in X_Train / X_Test
        
        line = list(map(str,i.split()))
        Y_T.append(float(line[-1]))
        
        for word in line[1:-2]:
            X_T[-1][0] += check_elongation(word)
            X_T[-1][1] += check_hashtag(word)
            X_T[-1][2] += check_CAPS(word)
            X_T[-1][3] += check_tag(word)
            X_T[-1][4] += check_negation(word)
            X_T[-1][5] += check_word_emotion(word)

        X_T[-1][6] +=  check_joy_hashtag(line)
        X_T[-1][7] +=  check_exp_emo(line)
        sentences.append(' '.join(line[1:-2]))
    
    VADER(X_T, sentences)   # appends X_T[-1][8:13], 4 new columns


# ## Load Training & Testing datasets
# * Load training & testing data
# * Call preprocess function on them
# * Make X_Train , Y_Train , X_Test & Y_Test

# In[6]:


########## load & preprocess files

with open(TRAIN_DATA_PATH, 'r') as f:
    TT_preprocess(f, Train_Sentence, Y_Train, X_Train)

with open(TEST_DATA_PATH, 'r') as f:
    TT_preprocess(f, Test_Sentence, Y_Test, X_Test)


print("Time to preprocess training & testing data ->",time() - start_time)


# ## Count Vectorization
# * Convert all X/Y_Train & X/Y_Test to DataFrames
# * Apply Count Vectorization on all Training & Testing sentences (unigram + bigram)
# * Convert it to pandas dataframe & concatenate it with X_Train & X_Test respectively

# In[7]:


####################### convert to dataframes

print("Making DF...")
X_Train_DF = pandas.DataFrame(X_Train)
Y_Train_DF = pandas.DataFrame(Y_Train)
X_Test_DF = pandas.DataFrame(X_Test)
Y_Test_DF = pandas.DataFrame(Y_Test)

####################### apply count vectorization

count_vectorizer = CountVectorizer(ngram_range=(1,2))    # Unigram and Bigram
Vectorized_Train = count_vectorizer.fit_transform(Train_Sentence)
Vectorized_Test = count_vectorizer.transform(Test_Sentence)

########### conver to DF & concat to X_Train & X_Test

# Convert sparse csr_matrix to dense format and allow columns to contain the array mapping from feature integer indices to feature names
count_vect_df = pandas.DataFrame(Vectorized_Train.todense(), columns=count_vectorizer.get_feature_names())
# Concatenate the original X_Train and the count_vect_df columnwise.
X_Train_DF = pandas.concat([X_Train_DF, count_vect_df], axis=1)

# Convert sparse csr_matrix to dense format and allow columns to contain the array mapping from feature integer indices to feature names
count_vect_df = pandas.DataFrame(Vectorized_Test.todense(), columns=count_vectorizer.get_feature_names())
# Concatenate the original X_Train and the count_vect_df columnwise.
X_Test_DF = pandas.concat([X_Test_DF, count_vect_df], axis=1)


print("Time to apply count Vectorization, make DFs & merge them ->",time() - start_time)


# ## Statistics function
# * Function that takes model predicted results & actual results
# * Compares them & print - <br>
# `Mean Absolute Error` <br>
# `Mean Squared Error` <br>
# `Root Mean Squared Error` <br>
# `R2 - Score` <br>
# `Pearson correlation, p-value` <br>
# `Spearman Result`
# 

# In[8]:


def get_statistics(result, Y_Test):
    MAE = metrics.mean_absolute_error(Y_Test, result)    
    MSE = metrics.mean_squared_error(Y_Test, result)     
    rmse = sqrt(MSE)
    r2 = metrics.r2_score(Y_Test, result)

    print("Results of sklearn.metrics:")
    print("MAE:",MAE)
    print("MSE:", MSE)
    print("RMSE:", rmse)
    print("R-Squared:", r2)
    print("\npearson corr. , p valve =",stats.pearsonr(Y_Test,result))
    print(stats.spearmanr(Y_Test,result))


# ## model_prediction function
# * Takes input as model classifier & the DataFrames
# * Prepares model & predict results
# * Calls get_statistics function for final evaluation

# In[9]:


def model_prediction(classifier, X_Train_DF, Y_Train, X_Test_DF, Y_Test):
    model = classifier
    model.fit(X_Train_DF, Y_Train)
    result = model.predict(X_Test_DF) 
    
    get_statistics(result, Y_Test)


# ## SVM

# In[10]:


## SVM
print("\nResult for SVM ->")
model_prediction(SVR(), X_Train_DF, Y_Train, X_Test_DF, Y_Test)
print("Time taken by SVM model ->",time() - start_time)


# ## Decision Tree Classifier

# In[11]:


## Using Decision Tree
print("\nResult Decision Tree ->")
model_prediction(DecisionTreeRegressor(max_depth = 5), X_Train_DF, Y_Train, X_Test_DF, Y_Test)
print("Time taken by Decision Tree ->",time() - start_time)


# ## MLP Regressor

# In[12]:


## using MLP
print("\nResult for MLP ->")
clf = MLPRegressor(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1,max_iter=1000)
model_prediction(clf, X_Train_DF, Y_Train, X_Test_DF, Y_Test)
print("Time taken by MLP ->",time() - start_time)


# In[ ]:





# In[ ]:




