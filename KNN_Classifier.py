#!/usr/bin/env python
# coding: utf-8

# In[45]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score


# In[18]:


df = pd.read_csv('malware.csv')


# In[19]:


df.head(5)


# In[20]:


#Getting required columns###
df_new = df[['classification', 'os', 'usage_counter', 'prio', 'static_prio', 'normal_prio', 'vm_pgoff', 'vm_truncate_count', 'task_size', 'map_count', 'hiwater_rss', 'total_vm', 'shared_vm', 'exec_vm', 'reserved_vm', 'nr_ptes', 'nvcsw', 'nivcsw','signal_nvcsw']]


# In[21]:


df.shape


# In[22]:


df_new.shape


# In[23]:


#checking for null values###
df_new.isnull().sum()


# In[24]:


df_new.head(5)


# In[25]:


df_new.dtypes


# In[26]:


df['os'].unique()


# In[27]:


#converting categorical variable to nominal
df_discretized=df_new.copy(deep=True)
os = df['os']
le = preprocessing.LabelEncoder()
le.fit(os)
os_encoded = le.transform(os)
print(os_encoded)
df_discretized['os'] = os_encoded


# In[28]:


df_discretized.head(5)


# In[29]:


#since some of the columns in data table have only 0's we removed them beacuse it will create problems during normalization### 
df_discretized1 = df_discretized.drop(['usage_counter','normal_prio','vm_pgoff','task_size','hiwater_rss','nr_ptes','signal_nvcsw'],axis=1)


# In[30]:


df_discretized1.head(5)


# In[31]:


#normalization of the data#
for col in df_discretized1.columns:
    if col != 'classification':
        df_discretized1[col]=(df_discretized1[col]-df_discretized1[col].min())/(df_discretized1[col].max()-df_discretized1[col].min())


# In[32]:


df_discretized1


# In[33]:


df_normalised = df_discretized1.copy(deep=True)


# In[34]:


df_normalised


# In[35]:


#label encoding the target variable and removing it from the feature table###
from sklearn import preprocessing
y = df_normalised['classification']
le = preprocessing.LabelEncoder()
le.fit(y)
y_encoded = le.transform(y) 

print(y_encoded)
df_normalised['classification'] = y_encoded
x_features = df_normalised.drop('classification',axis=1)


# In[36]:


# splitting the data into test and train with training as 75% and test data as 25%#######
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_features, y_encoded, 
                                                    test_size=0.25, random_state = 42, stratify=None)


# In[37]:


x_train.shape


# In[38]:


x_test.shape


# In[39]:


y_train.shape


# In[50]:


is_contiguous = np.ascontiguousarray(x_test).flags.c_contiguous
print(is_contiguous)


# In[46]:


from sklearn import neighbors
for k in range(1,1000,100): 
    clf=neighbors.KNeighborsClassifier(n_neighbors=k, metric='cosine')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print('K =', k, ', Accuracy: ', accuracy_score(y_test, y_pred), ', Precision: ', precision_score(y_test, y_pred, average='macro', zero_division=0),
         ', Recall: ', recall_score(y_test, y_pred, average='macro', zero_division=0))


# In[306]:


neighbors = np.arange(1, 9)
training_accuracy = np.empty(len(neighbors))
testing_accuracy = np.empty(len(neighbors))
  
for i, k in enumerate(neighbors):
    classifier = KNeighborsClassifier(n_neighbors=k, metric ='cosine')
    classifier.fit(x_train, y_train)
      
    training_accuracy[i] = classifier.score(x_train, y_train)
    testing_accuracy[i] = classifier.score(x_test, y_test)
  
plt.plot(neighbors, testing_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, training_accuracy, label = 'Training Accuracy')
  
plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()


# In[311]:


from sklearn import neighbors
for k in range(1,1000,100): 
    clf=neighbors.KNeighborsClassifier(n_neighbors=k, metric='manhattan')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print('K =', k, ', Accuracy: ', accuracy_score(y_test, y_pred), ', Precision: ', precision_score(y_test, y_pred, average='macro', zero_division=0),
         ', Recall: ', recall_score(y_test, y_pred, average='macro', zero_division=0))


# In[313]:


from sklearn import neighbors
for k in range(1,1000,100): 
    clf=neighbors.KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print('K =', k, ', Accuracy: ', accuracy_score(y_test, y_pred), ', Precision: ', precision_score(y_test, y_pred, average='macro', zero_division=0),
         ', Recall: ', recall_score(y_test, y_pred, average='macro', zero_division=0))


# In[ ]:


# We have choosen values of k from 1 to 1000 with 100 interval because we have a large dataset comprising og 100k datapoints
and we have choosen diffrent paramaters comprising of different distances like eucledian, manhattan, cosine and we see accuracies decreasing from 
99 to 89 for all the distances which is a sign for reducing overfitting####


# In[314]:


from sklearn.model_selection import cross_val_score
k1 = [200, 300, 500, 750, 1000]
for k in k1: 
    clf=neighbors.KNeighborsClassifier(k, metric='euclidean')
    acc=cross_val_score(clf, x_features, y_encoded, cv=5, scoring='accuracy').mean()
    print('K =', k, ', Accuracy: ',acc)


# In[315]:


k1 = [200, 300, 500, 750, 1000]
for k in k1: 
    clf=neighbors.KNeighborsClassifier(k, metric='manhattan')
    acc=cross_val_score(clf, x_features, y_encoded, cv=5, scoring='accuracy').mean()
    print('K =', k, ', Accuracy: ',acc)


# In[ ]:


# We have choosen values of k [200, 300, 500, 750, 1000] because we have a large dataset comprising og 100k datapoints and cross_val_score with n as 5
and we have choosen diffrent paramaters comprising of different distances like eucledian, manhattan and we have seen some accuracies starting from 
79 to 80 for both the distances####

