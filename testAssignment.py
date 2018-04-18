
# coding: utf-8

# Import Libraries

# In[1]:


import scipy.io
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix


# Importing sklearn libraries for Machine learning.

# In[2]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score


# In[3]:


# loading the matlab file

# In[4]:


trainingFeatures = pd.read_csv('training2.csv', header=None);


# In[5]:


testFeatures = pd.read_csv('test2.csv', header=None)


# In[6]:


trainingClassifier = pd.read_csv('ctrain.csv', header=None)


# In[7]:


trainingFeatures.head()


# In[8]:


testFeatures.head()


# In[9]:


#Training Features
features = trainingFeatures

#Training Classifier
classifier = trainingClassifier

#Test Features
test = testFeatures
#need to predict the classifier for test


# In[10]:

print("read")
#Training the model using logistic regression
print("step1")
#scaler = StandardScaler()
#X_train_scaled = scaler.fit_transform(features)
print("step2")
svmmodel = SVC()
svmmodel.fit(features, classifier)
print("step3")
#X_test_scaled = scaler.transform(features)
print("step4")


# ## Testing the Model
# Now its time to perdict the 'Classifications' our our Test Model, given the Test Features

# In[11]:


predict = svmmodel.predict(test)
print("success")
print(predict)


# In[13]:


# Write 'perdict' to a csv file
df = pd.DataFrame(
{
    'class': predict
})
df.index = df.index + 1
df.to_csv('submit.csv')

