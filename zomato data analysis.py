#!/usr/bin/env python
# coding: utf-8

# # Zomato data analysis project

# # Step 1 - Importing Libraries

# In[ ]:


pandas is used for data manipulation and analysis.
numpy is used for numerical operations.
matplotlib.pyplot and seaborn are used for data visualization.


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Step 2 - Create the data frame

# In[6]:


dataframe = pd.read_csv("Zomato data .csv")
print(dataframe)


# In[7]:


dataframe


# # Convert the data type of column - rate

# In[11]:


def handleRate(value):
    value = str(value).split('/')
    value = value[0];
    return float(value)

dataframe['rate'] = dataframe['rate'].apply(handleRate)
print(dataframe.head())


# In[13]:


dataframe.info()


# # Type of resturant

# In[15]:


dataframe.head()


# In[16]:


sns.countplot(x=dataframe['listed_in(type)'])
plt.xlabel("type of resturant")


# # conclusion - majority of resturant falls in dining category

# In[18]:


dataframe.head()


# In[26]:


plt.hist(dataframe['rate'],bins =5)
plt.title("ratings distribution")
plt.show()


# # conclusion - the majority resturants received ratings from 3.5 to 4.

# # Average order spending by couples

# In[27]:


dataframe.head()


# In[28]:


couple_data=dataframe['approx_cost(for two people)']
sns.countplot(x=couple_data)


# # conclusion - the majority of couples preferr resturants with an approximate cost of 300 rupees.

# # which mode receives maximum rating

# In[29]:


dataframe.head()


# In[30]:


plt.figure(figsize = (6,6))
sns.boxplot(x = 'online_order', y = 'rate', data = dataframe)


# # conclusion - offline order receive lower rating in comparison to online order

# In[33]:


pivot_table = dataframe.pivot_table(index='listed_in(type)', columns='online_order', aggfunc='size', fill_value=0)
sns.heatmap(pivot_table, annot=True, cmap="Accent", fmt='d')
plt.title("Heatmap")
plt.xlabel("online order")
plt.ylabel("Listed In (Type)")
plt.show()


# # Dining restaurants primarily accept offline orders, whereas cafes primarily receive online orders. This suggests that clients preferred orders in person at restaurants, but prefer online ordering at cafes. 

# In[ ]:




