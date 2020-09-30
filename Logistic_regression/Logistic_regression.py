#!/usr/bin/env python
# coding: utf-8

# # Regularization with logestic regression
Regularization helps to solve over fitting problem in machine learning. Simple model will be a very poor generalization of data. At the same time, complex model may not perform well in test data due to over fitting. We need to choose the right model in between simple and complex model. Regularization helps to choose preferred model complexity, so that model is better at predicting. Regularization is nothing but adding a penalty term to the objective function and control the model complexity using that penalty term. It can be used for many machine learning algorithms.


#  ## Regularization of linear models
Regularization is a method for “constraining” or “regularizing” the size of the coefficients, thus “shrinking” them towards zero.
It reduces model variance and thus minimizes overfitting.
If the model is too complex, it tends to reduce variance more than it increases bias, resulting in a model that is more likely to generalize.

Our aim is to locate the optimum model complexity, and thus regularization is useful when we believe our model is too complex.
# ## Logestic regression without regularization

# In[30]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[31]:


breast_cancer=pd.read_csv(r"D:\msc3\machine learning\lab7\data.csv")
breast_cancer.head(10)


# In[32]:


print("Number of data:"+str(len(breast_cancer.index)))


# In[33]:


breast_cancer.info()


# In[34]:


breast_cancer.isnull().any()


# # Analyzing data

# In[35]:


sns.countplot("diagnosis",data=breast_cancer)


# In[36]:


breast_cancer.diagnosis.value_counts()


# In[37]:


breast_cancer.hist(bins=10,figsize=(20,20),grid=False)


# In[38]:


X=breast_cancer.drop("diagnosis",axis=1)
y=breast_cancer["diagnosis"]


# In[39]:


features_mean= list(breast_cancer.columns[2:22])


# In[40]:


plt.figure(figsize=(20,20))
sns.heatmap(breast_cancer[features_mean].corr(), annot=True, square=True, cmap='coolwarm')
plt.show()


# In[41]:


breast_cancer.drop("id", axis=1, inplace=True)


# In[42]:


breast_cancer.head(10)


# In[43]:


diagno=pd.get_dummies(breast_cancer['diagnosis'], drop_first=True)
diagno.head()


# In[44]:


breast_cancer=pd.concat((breast_cancer,diagno),axis=1)


# In[45]:


breast_cancer.info()


# In[46]:


breast_cancer.drop("diagnosis", axis=1, inplace=True)


# In[47]:


breast_cancer.head()


# In[48]:


X=breast_cancer.drop("M",axis=1)
y=breast_cancer["M"]


# In[49]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=2)


# In[50]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1e5)
logreg.fit(X_train,y_train)
predictions=logreg.predict(X_test)


# In[51]:


from sklearn.metrics import classification_report
classification_report(y_test,predictions)


# In[52]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)


# In[53]:


from sklearn.metrics import accuracy_score
print("train accuracy:")
print(format(logreg.score(X_train,y_train)*100.0))
print("test accuracy:")
accuracy_score(y_test,predictions)


# In[54]:


#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['features','effect_score']  #naming the dataframe columns
print(featureScores.nlargest(10,'effect_score'))  #print 10 best features


# In[82]:


from sklearn.model_selection import cross_val_score
mse=cross_val_score (logreg,X_test,y_test,scoring='neg_mean_squared_error',cv=10)
mean_mse=np.mean(mse)
print(mean_mse)


# In[58]:


from sklearn.metrics import r2_score
from sklearn import metrics
from sklearn.linear_model import Ridge
import warnings 
warnings.filterwarnings('ignore')
from sklearn.linear_model import Ridge
ridgereg = Ridge(alpha=0, normalize=True)
ridgereg.fit(X_train, y_train)
y_pred = ridgereg.predict(X_test)
print("R-Square Value",r2_score(y_test,y_pred))
print("\n")
print ("mean_absolute_error :",metrics.mean_absolute_error(y_test, y_pred))
print("\n")
print ("mean_squared_error : ",metrics.mean_squared_error(y_test, y_pred))
print("\n")
print ("root_mean_squared_error : ",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# ## Ridge Regularization

# In[83]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
ridge=Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
ridge_regression=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_regression.fit(X_train, y_train)


# In[84]:


print(ridge_regression.best_params_)
print(ridge_regression.best_score_)


# In[85]:



ridgereg = Ridge(0.01, normalize=True)
ridgereg.fit(X_train, y_train)
ridge_pred = ridgereg.predict(X_test)
print("R-Square Value",r2_score(y_test,ridge_pred))
print("\n")
print ("mean_absolute_error :",metrics.mean_absolute_error(y_test, y_pred))
print("\n")
print ("mean_squared_error : ",metrics.mean_squared_error(y_test, y_pred))
print("\n")
print ("root_mean_squared_error : ",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[86]:


sns.distplot(y_test-ridge_pred)


# ## Lasso Regularization

# In[87]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
lasso=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
lasso_regression=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)
lasso_regression.fit(X_train, y_train)
print(lasso_regression.best_params_)
print(lasso_regression.best_score_)


# In[88]:


lassoreg = Lasso(1e-08, normalize=True)
lassoreg.fit(X_train, y_train)
lasso_pred = lassoreg.predict(X_test)
print("R-Square Value",r2_score(y_test,lasso_pred))
print("\n")
print ("mean_absolute_error :",metrics.mean_absolute_error(y_test, y_pred))
print("\n")
print ("mean_squared_error : ",metrics.mean_squared_error(y_test, y_pred))
print("\n")
print ("root_mean_squared_error : ",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[89]:


sns.distplot(y_test-lasso_pred)


# In[ ]:





# In[ ]:




