#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


# In[19]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# In[20]:


iris=load_iris()
irisdata=pd.DataFrame(iris.data,columns=iris.feature_names)
print(iris.feature_names)
irisdata['target']=iris.target


# In[21]:


print(irisdata.head())


# In[22]:


import seaborn as sns
sns.set({'figure.figsize':(5,5)})
sns.countplot(irisdata['target'])


# In[23]:


irisdata.target.value_counts()


# In[24]:


training_x=iris.data
training_y=iris.target
x_train,x_test,y_train,y_test=train_test_split(training_x,training_y,test_size=0.2)
print("len_of_training_dataset:",len(x_train))
print("len_of_testing_dataset:",len(x_test))


# In[25]:


model=DecisionTreeClassifier()
model.fit(x_train,y_train)
model.score(x_test,y_test)
y_pred=model.predict(x_test)
print(y_pred)
print(y_test)


# In[26]:


print("accuracy_score:",accuracy_score(y_pred,y_test))
#label=[0,1,2]
print("confusion_matrix 1:",confusion_matrix(y_pred,y_test))
print("classification_report:",classification_report(y_pred,y_test))


# In[27]:


fig,axes=plt.subplots(nrows=1,ncols=1,figsize=(10,10),dpi=500)
tree.plot_tree(model,feature_names=iris.feature_names,class_names=iris.target_names)

