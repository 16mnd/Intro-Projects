#!/usr/bin/env python
# coding: utf-8

# <img src="https://rhyme.com/assets/img/logo-dark.png" align="center"> <h2 align="center">Logistic Regression: A Sentiment Analysis Case Study</h2>

#  

#  

# ### Introduction
# ___

# - IMDB movie reviews dataset
# - http://ai.stanford.edu/~amaas/data/sentiment
# - Contains 25000 positive and 25000 negative reviews
# <img src="https://i.imgur.com/lQNnqgi.png" align="center">
# - Contains at most reviews per movie
# - At least 7 stars out of 10 $\rightarrow$ positive (label = 1)
# - At most 4 stars out of 10 $\rightarrow$ negative (label = 0)
# - 50/50 train/test split
# - Evaluation accuracy

# <b>Features: bag of 1-grams with TF-IDF values</b>:
# - Extremely sparse feature matrix - close to 97% are zeros

#  <b>Model: Logistic regression</b>
# - $p(y = 1|x) = \sigma(w^{T}x)$
# - Linear classification model
# - Can handle sparse data
# - Fast to train
# - Weights can be interpreted
# <img src="https://i.imgur.com/VieM41f.png" align="center" width=500 height=500>

# ### Task 1: Loading the dataset
# ---

# In[38]:


import pandas as pd

df = pd.read_csv('data/movie_data.csv')
df.head(10)


# In[39]:


df['review'][0]


# ## <h2 align="center">Bag of words / Bag of N-grams model</h2>

# ### Task 2: Transforming documents into feature vectors

# Below, we will call the fit_transform method on CountVectorizer. This will construct the vocabulary of the bag-of-words model and transform the following three sentences into sparse feature vectors:
# 1. The sun is shining
# 2. The weather is sweet
# 3. The sun is shining, the weather is sweet, and one and one is two
# 

# In[40]:


import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer()

docs=np.array(['The sun is shining',
'The weather is sweet',
'The sun is shining, the weather is sweet, and one and one is two'])
bag= count.fit_transform(docs)


# In[41]:


print(count.vocabulary_)


# In[42]:


print(bag.toarray())


# Raw term frequencies: *tf (t,d)*—the number of times a term t occurs in a document *d*

# ### Task 3: Word relevancy using term frequency-inverse document frequency

# $$\text{tf-idf}(t,d)=\text{tf (t,d)}\times \text{idf}(t,d)$$

# $$\text{idf}(t,d) = \text{log}\frac{n_d}{1+\text{df}(d, t)},$$

# where $n_d$ is the total number of documents, and df(d, t) is the number of documents d that contain the term t.

# In[43]:


from sklearn.feature_extraction.text import TfidfTransformer
np.set_printoptions(precision=2)
tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())


# In[ ]:





# The equations for the idf and tf-idf that are implemented in scikit-learn are:
# 
# $$\text{idf} (t,d) = log\frac{1 + n_d}{1 + \text{df}(d, t)}$$
# The tf-idf equation that is implemented in scikit-learn is as follows:
# 
# $$\text{tf-idf}(t,d) = \text{tf}(t,d) \times (\text{idf}(t,d)+1)$$

# ### Task 4: Data Preparation

# In[44]:


df.loc[0, 'review'][-50:]


# In[45]:


import re
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) +        ' '.join(emoticons).replace('-', '')
    return text


# In[46]:


preprocessor(df.loc[0,'review'][-50:])


# In[47]:


df['review']=df['review'].apply(preprocessor)


# In[ ]:





#  

#  

# ### Task 5: Tokenization of documents

# In[48]:


from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()


# In[49]:


def tokenizer(text):
    return text.split()


# In[50]:


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


# In[51]:


tokenizer('runners like running and thus they run')


# In[52]:


tokenizer_porter('runners like running and thus they run')


# In[55]:


import nltk
nltk.download('stopwords')


#  

# In[56]:


from nltk.corpus import stopwords 
stop= stopwords.words('english')
[w for w in tokenizer_porter('a runner likes runing and runs a lot')[-10:] if w not in stop]


#  

# ### Task 6: Transform Text Data into TF-IDF Vectors

# In[57]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(strip_accents=None,lowercase=False, 
                        preprocessor=None, 
                        tokenizer=tokenizer_porter,
                        use_idf= True, 
                        norm='l2',
                        smooth_idf=True)
y= df.sentiment.values
x= tfidf.fit_transform(df.review)


# In[ ]:





# ### Task 7: Document Classification using Logistic Regression

# In[58]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(x,y, random_state=1, 
                                                  test_size=0.5, 
                                                  shuffle=False)


# In[59]:


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


# ### Task 8: Model Evaluation

# In[61]:


filename='saved_model.sav'
saved_clf=pickle.load(open(filename,'rb'))


# In[62]:


saved_clf.score(X_test, y_test)


# In[ ]:




