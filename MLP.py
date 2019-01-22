
# coding: utf-8

# In[9]:


import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np


# In[10]:


df = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv')


# In[20]:


attr = df.values
print(df.columns.values)


# In[37]:


df_x = attr[:, [4,5]]
df_y = attr[:, -1]
print(df_x)


# In[34]:


from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(df_x, df_y, test_size = 0.2, random_state = 42)


# In[40]:


print(X_train.shape)


# In[41]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()


# In[44]:


model.fit(X_train, y_train)
np.set_printoptions(precision = 4, suppress = True)
print(model.coef_, '%.4f' % model.intercept_)


# In[45]:


y_pred = model.predict(X_valid)


# In[47]:


y_pred_train = model.predict(X_train)


# In[49]:


e = y_valid - y_pred


# In[53]:


mae = np.mean(np.abs(e))
print('%.4f' % mae)

