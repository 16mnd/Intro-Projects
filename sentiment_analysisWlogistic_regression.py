

# - IMDB movie reviews dataset
# - http://ai.stanford.edu/~amaas/data/sentiment
# - Contains 25000 positive and 25000 negative reviews
# - Contains at most reviews per movie
# - At least 7 stars out of 10  (label = 1)
# - At most 4 stars out of 10 (label = 0)
# - 50/50 train/test split
# - Evaluation accuracy

# - Features: bag of 1-grams with TF-IDF values</b>:
# - Extremely sparse feature matrix - close to 97% are zeros

# - Model: Logistic regression
# - (y = 1|x) = sigma(w^{T}x)
# - Linear classification model
# - Can handle sparse data
# - Fast to train
# - Weights can be interpreted


# ### Task 1: Loading the dataset
import pandas as pd

df = pd.read_csv('data/movie_data.csv')
df.head(10)



df['review'][0] #reviewing a singular review



# ### Task 2: Transforming documents into feature vectors

# Below, we will call the fit_transform method on CountVectorizer. This will construct the vocabulary of the bag-of-words model and transform the following three sentences into sparse feature vectors:
# 1. The sun is shining
# 2. The weather is sweet
# 3. The sun is shining, the weather is sweet, and one and one is two
# 




import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer()

#demo run of 
docs=np.array(['The sun is shining',
'The weather is sweet',
'The sun is shining, the weather is sweet, and one and one is two'])
bag= count.fit_transform(docs)


print(count.vocabulary_) #returns labels of words



print(bag.toarray())     #the actual bag of words returned 


# Raw term frequencies: tf-(t,d): the number of times a term t occurs in a document *d*

# ### Task 3: Word relevancy using term frequency-inverse document frequency

# where n_d is the total number of documents, and df(d, t) is the number of documents d that contain the term t.



from sklearn.feature_extraction.text import TfidfTransformer
np.set_printoptions(precision=2)
tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())








# The equations for the idf and tf-idf that are implemented in scikit-learn are:
# 
# text{idf} (t,d) = log ( (1 + n_d)/(1 + (df)(d, t))
# The tf-idf equation that is implemented in scikit-learn is as follows:
# 
# text{tf-idf}(t,d) = text{tf}(t,d) * ((idf)(t,d)+1)

# ### Task 4: Data Preparation




df.loc[0, 'review'][-50:] #examination of first review, more specifically the first reviews last 50 words


## gets rid of unnecessary lexicons and emojis
import re
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) +        ' '.join(emoticons).replace('-', '')
    return text





preprocessor(df.loc[0,'review'][-50:])




df['review']=df['review'].apply(preprocessor)





# ### Task 5: Tokenization of documents


from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()




def tokenizer(text):
    return text.split()





def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]




#random example of tokenizer by itself
tokenizer('runners like running and thus they run')


#example of the stemmer tokenizer
tokenizer_porter('runners like running and thus they run')





import nltk
nltk.download('stopwords')


from nltk.corpus import stopwords 
stop= stopwords.words('english')
[w for w in tokenizer_porter('a runner likes runing and runs a lot')[-10:] if w not in stop]


# Task 6: Transform Text Data into TF-IDF Vectors


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(strip_accents=None,lowercase=False, 
                        preprocessor=None, 
                        tokenizer=tokenizer_porter,
                        use_idf= True, 
                        norm='l2',
                        smooth_idf=True)
y= df.sentiment.values
x= tfidf.fit_transform(df.review)



# Task 7: Document Classification using Logistic Regression


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(x,y, random_state=1, 
                                                  test_size=0.5, 
                                                  shuffle=False)





import pickle
from sklearn.linear_model import LogisticRegressionCV

clf= LogisticRegressionCV(cv=5,scoring='accuracy',
                          random_state=0,
                          n_jobs=-1, 
                          verbose=3,
                          max_iter=300).fit(X_train, y_train)
saved_model=open('saved_model.sav', 'wb')
pickle.dump(clf, saved_model)
saved_model.close()



#saving model status (cross validation(epoch), weights, etc) just incase

filename='saved_model.sav'
saved_clf=pickle.load(open(filename,'rb'))





saved_clf.score(X_test, y_test)







