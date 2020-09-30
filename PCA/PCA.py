#!/usr/bin/env python
# coding: utf-8

# In[57]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from sklearn.decomposition import PCA
import seaborn as sns
sns.set()
df=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", names=['sepal length','sepal width','petal length','petal width','target'])
print(df)


# In[58]:


features = ['sepal length', 'sepal width', 'petal length', 'petal width']
# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['target']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)
print(x)


# In[59]:


pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['component1', 'component2'])
    


# In[60]:



df.describe() 


# In[61]:


plt.figure(figsize = (10, 7)) 
a = df["sepal length"] 

plt.hist(a, bins = 20, color = "green") 
plt.title("Sepal Length in cm") 
plt.xlabel("Sepal_Length_cm") 
plt.ylabel("Count") 


# In[62]:


plt.figure(figsize = (10, 7)) 
b = df["sepal width"] 

plt.hist(b, bins = 20, color = "green") 
plt.title("sepal width in cm") 
plt.xlabel("sepal_width_cm") 
plt.ylabel("Count") 


# In[63]:


plt.figure(figsize = (10, 7)) 
c = df["petal length"] 

plt.hist(c, bins = 20, color = "green") 
plt.title("petal length in cm") 
plt.xlabel("petal_length_cm") 
plt.ylabel("Count") 


# In[64]:


plt.figure(figsize = (10, 7)) 
d = df["petal width"] 

plt.hist(d, bins = 20, color = "green") 
plt.title("petal width in cm") 
plt.xlabel("petal_width_cm") 
plt.ylabel("Count") 


# In[65]:


plt.figure(figsize=(15,15))
p=sns.heatmap(df.corr(), annot=True,cmap='RdYlGn')


# In[52]:


print(principalDf)


# In[40]:


finalDf = pd.concat([principalDf, df[['target']]], axis = 1)


# In[41]:


print(finalDf)


# In[42]:


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('component1', fontsize = 15)
ax.set_ylabel('component2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'component1']
               , finalDf.loc[indicesToKeep, 'component2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()


# In[34]:


pca.explained_variance_ratio_


# In[66]:


plt.figure(figsize=(15,15))
p=sns.heatmap(finalDf.corr(), annot=True,cmap='RdYlGn')


# In[ ]:




