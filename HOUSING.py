#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from IPython.core.display import display
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
'exec(%matplotlib inline)'

data = pd.read_csv('HOUSING.csv')


# In[22]:


data.describe()


# In[23]:



data.head()


# In[28]:


data.columns


# In[35]:


X=data[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','sqft_above','sqft_basement','yr_built','yr_renovated']]
y=data[['price']]


# In[36]:


model=LinearRegression()


# In[37]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)


# In[38]:


model.fit(X_train, y_train)


# In[45]:


X = data.values
y_pred = model.predict(X_test)
plt.scatter(y_test,y_pred)


# In[ ]:




