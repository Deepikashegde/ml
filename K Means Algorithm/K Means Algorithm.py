#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[31]:


train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)
test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)


# In[32]:


print("***** Train_Set *****")
print(train.head())


# In[33]:


print("***Test set***")
print(test.head())


# In[34]:


print("***** Train_Set *****")
print(train.describe())


# In[35]:


print("***test set***")
print(test.describe())


# In[36]:


train.isna().head()


# In[37]:


test.isna().head()


# In[38]:


print("*****In the train set*****")
print(train.isna().sum())


# In[39]:


train.fillna(train.mean(), inplace=True)


# In[40]:


test.fillna(train.mean(), inplace=True)


# In[41]:


print(train.isna().sum())


# In[42]:


print(test.isna().sum())


# In[43]:


train['Ticket'].head()


# In[44]:


train['Cabin'].head()


# In[45]:


train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[46]:


train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[47]:


g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=20)


# In[48]:


grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();


# In[49]:


train = train.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)
test = test.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)


# In[50]:


labelEncoder = LabelEncoder()
labelEncoder.fit(train['Sex'])
labelEncoder.fit(test['Sex'])
train['Sex'] = labelEncoder.transform(train['Sex'])
test['Sex'] = labelEncoder.transform(test['Sex'])


# In[51]:


train.info()


# In[52]:


test.info()


# In[53]:


X = np.array(train.drop(['Survived'], 1).astype(float))


# In[54]:


y = np.array(train['Survived'])


# In[55]:


train.info()


# In[56]:


kmeans = KMeans(n_clusters=2) 
kmeans.fit(X)


# In[57]:


correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(X))


# In[58]:


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# In[59]:


kmeans.fit(X_scaled)


# In[60]:


correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(X))


# In[68]:


K=range(len(X))
plt.plot(K,prediction)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('predicted')
plt.show()


# In[ ]:




