#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
#!pip install name 


# In[2]:


data = sns.load_dataset("titanic")
data.head()


# In[3]:


data.dropna(inplace =True)


# In[4]:


target = "survived"
features = data.drop(target, axis=1)


# In[5]:


data_encoded = pd.get_dummies(data, drop_first = True)


# In[6]:


X = data_encoded.drop(target, axis=1)#features
y =data_encoded[target]


# In[7]:


X_train,X_test, y_train, y_test =train_test_split(X, y, test_size =0.2, random_state = 42)


# In[8]:


classifier = DecisionTreeClassifier()


# In[9]:


classifier.fit(X_train, y_train)


# In[16]:


#y=f(x)
y_pred = classifier.predict(X_test)


# In[17]:


print(y_pred)
print(y_test)


# In[22]:


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[24]:




