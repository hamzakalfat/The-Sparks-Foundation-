#!/usr/bin/env python
# coding: utf-8

# SUPERVISED ML- Student Score Prediction - Linear Regression
HAMZA KALFAT
# #  Import librairies

# In[11]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
get_ipython().run_line_magic('matplotlib', 'inline')


# # Import Data

# In[17]:


url = "http://bit.ly/w-data"
s_data = pd.read_csv(url)


# # Data checking

# In[18]:


s_data.describe()


# In[14]:


print("Number of raws :" +str(s_data.shape[0]))
print ("Number of columns :"+str(s_data.shape[1]))


# In[15]:


s_data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# # Training the Algorithm

#     1) linearity check

# In[19]:


X = s_data.iloc[:, :-1].values  
y = s_data.iloc[:, 1].values  


# In[37]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
LR=LinearRegression()
LR.fit(X_train,y_train)


# **plotting the regression line using the formula y=m*x+c

# In[38]:


m=LR.coef_
c=LR.intercept_
line=m*X+c
plt.scatter(X,y)
plt.plot(X,line)
plt.show()


# # Making predictions

# In[39]:


PR=LR.predict(X_test)
list(zip(y_test,PR))


# # Evaluating the model

# In[40]:


from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,PR,squared=False)


# # Solution

# In[41]:


hour =[9.25]
own_pr=LR.predict([hour])
print("No of Hours = {}".format([hour]))
print("Predicted Score = {}".format(own_pr[0]))


# 
