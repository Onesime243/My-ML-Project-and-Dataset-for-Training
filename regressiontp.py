#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


data = pd.read_csv("D:\\joao\IO\\python\\Machine Learning & Python\\StudentPerformanceFactors.csv")


# In[4]:


data.head()


# In[5]:


data.dtypes


# In[6]:


data.info()


# In[41]:


data['Parental_Involvement'] = data['Parental_Involvement'].replace({'Low': 0, 'Medium': 1,'High': 2})


# In[9]:


data['Access_to_Resources'] = data['Access_to_Resources'].replace({'Medium': 0, 'High': 1})


# In[10]:


data['Extracurricular_Activities'] = data['Extracurricular_Activities'].replace({'no': 0, 'yes': 1})


# In[11]:


data['Motivation_Level'] = data['Motivation_Level'].replace({'Low': 0, 'Medium': 1})


# In[12]:


data['Internet_Access'] = data['Internet_Access'].replace({'no': 0, 'yes': 1})


# In[13]:


data['Family_Income'] = data['Family_Income'].replace({'Low': 0, 'Medium': 1})


# In[15]:


data['Teacher_Quality'] = data['Teacher_Quality'].replace({'Medium': 0, 'High': 1})


# In[16]:


data['School_Type'] = data['School_Type'].replace({'Public': 0, 'Private': 1})


# In[17]:


data['Peer_Influence'] = data['Peer_Influence'].replace({'Neutral': 0, 'Positive': 1,'Negative':-1})


# In[18]:


data['Learning_Disabilities'] = data['Learning_Disabilities'].replace({'no': 0, 'yes': 1})


# In[19]:


data['Parental_Education_Level'] = data['Parental_Education_Level'].replace({'High School': 2, 'College': 1,'Postgraduate':3})


# In[20]:


data['Distance_from_Home'] = data['Distance_from_Home'].replace({'Near': 1, 'Moderate': 2})


# In[21]:


data['Gender'] = data['Gender'].replace({'Female': 0, 'Male': 1})


# In[22]:


data['Motivation_Level']= data['Motivation_Level'].astype('category')
data['Motivation_Level'] = data['Motivation_Level'].cat.codes


# In[23]:


data


# In[24]:


data['Access_to_Resources']= data['Access_to_Resources'].astype('category')
data['Access_to_Resources'] = data['Access_to_Resources'].cat.codes


# In[25]:


data


# In[26]:


data['Extracurricular_Activities']= data['Extracurricular_Activities'].astype('category')
data['Extracurricular_Activities'] = data['Extracurricular_Activities'].cat.codes


# In[27]:


data


# In[28]:


data['Motivation_Level']= data['Motivation_Level'].astype('category')
data['Motivation_Level'] = data['Motivation_Level'].cat.codes


# In[29]:


data['Internet_Access']= data['Internet_Access'].astype('category')
data['Internet_Access'] = data['Internet_Access'].cat.codes


# In[30]:


data['Family_Income']= data['Family_Income'].astype('category')
data['Family_Income'] = data['Family_Income'].cat.codes


# In[31]:


data['Teacher_Quality']= data['Teacher_Quality'].astype('category')
data['Teacher_Quality'] = data['Teacher_Quality'].cat.codes


# In[32]:


data['School_Type']= data['School_Type'].astype('category')
data['School_Type'] = data['School_Type'].cat.codes


# In[33]:


data['Peer_Influence']= data['Peer_Influence'].astype('category')
data['Peer_Influence'] = data['Peer_Influence'].cat.codes


# In[35]:


data['Learning_Disabilities']= data['Learning_Disabilities'].astype('category')
data['Learning_Disabilities'] = data['Learning_Disabilities'].cat.codes


# In[36]:


data['Parental_Education_Level']= data['Parental_Education_Level'].astype('category')
data['Parental_Education_Level'] = data['Parental_Education_Level'].cat.codes


# In[37]:


data['Distance_from_Home']= data['Distance_from_Home'].astype('category')
data['Distance_from_Home'] = data['Distance_from_Home'].cat.codes


# In[38]:


data['Gender']= data['Gender'].astype('category')
data['Gender'] = data['Gender'].cat.codes


# In[43]:


data['Parental_Involvement']= data['Parental_Involvement'].astype('category')
data['Parental_Involvement'] = data['Parental_Involvement'].cat.codes


# In[44]:


data


# In[46]:


corr = data.corr()
corr.shape


# In[47]:


corr


# In[48]:


data.isnull().sum()


# In[49]:


data.columns


# In[50]:


x = data.drop(columns = 'Exam_Score')
x


# In[51]:


y= data['Exam_Score']


# In[54]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=10, random_state=0)


# In[53]:


from sklearn.linear_model import LinearRegression


# In[70]:


lr_2 = LinearRegression()


# In[71]:


lr_2.fit(x_train, y_train)


# In[72]:


LinearRegression()


# In[73]:


c=lr_2.coef_
c


# In[74]:


c = lr_2.intercept_
c


# In[75]:


y_pred_train = lr_2.predict(x_train)


# In[76]:


y_pred_train


# In[77]:


import matplotlib.pyplot as plt
plt.scatter(y_train, y_pred_train)
plt.xlabel('Actual Exam_Score  ')
plt.ylabel('predict Exam_Score')
plt.show()


# In[78]:


y_train.to_csv('y_train.csv')
y_test.to_csv('y_test.csv')


# In[ ]:




