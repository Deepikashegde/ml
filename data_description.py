#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import seaborn as sns
sns.set()
breast_cancer = load_breast_cancer()

X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)


# In[2]:


print (X)


# In[3]:


dir(breast_cancer)


# In[4]:


y=pd.Categorical.from_codes(breast_cancer.target,breast_cancer.target_names)
print(y)


# In[6]:


#We will do this using SciKit-Learn library in Python using the train_test_split method.
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[7]:


#Feature Scaling to bring attribute to one range (say 0-100 or 0-1)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[8]:


print(X_train)
print(X_test)


# In[11]:


print("Cancer data set dimensions : {}".format(X.shape,y.shape))


# In[40]:


df=pd.read_csv(r"D:\msc3\machine learning\lab1\data.csv")
sns.boxplot(x='diagnosis', y='area_mean', data=df)


# In[41]:


malignant = df[df['diagnosis']=='M']['area_mean']
benign = df[df['diagnosis']=='B']['area_mean']
fig = plt.figure()
ax = fig.add_subplot(111)
ax.boxplot([malignant,benign], labels=['M', 'B'])


# In[42]:


malignant = df[df['diagnosis']=='M']['radius_mean']
benign = df[df['diagnosis']=='B']['radius_mean']
fig = plt.figure()
ax = fig.add_subplot(111)
ax.boxplot([malignant,benign], notch = True, labels=['M', 'B']);


# In[43]:


sns.boxplot(x='diagnosis', y='radius_mean', data=df)


# In[44]:


sns.boxplot(x='diagnosis', y='perimeter_mean', data=df)


# In[46]:


sns.FacetGrid(df, hue='diagnosis', height=5)  .map(sns.distplot, 'perimeter_mean')  .add_legend()
plt.show()


# In[47]:


sns.FacetGrid(df, hue='diagnosis', height=5)  .map(sns.distplot, 'radius_mean')  .add_legend()
plt.show()


# In[48]:


sns.FacetGrid(df, hue='diagnosis', height=5)  .map(sns.distplot, 'area_mean')  .add_legend()
plt.show()


# In[ ]:




