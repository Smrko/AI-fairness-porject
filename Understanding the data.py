#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
# data manipulation libraries
import pandas as pd
import numpy as np

from time import time

# Graphs libraries
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.style.use('seaborn-white')
import seaborn as sns
get_ipython().system('pip install ipywidgets')


# In[2]:


import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
from plotly import tools


# In[3]:


# Design libraries
from IPython.display import Markdown, display
import warnings
warnings.filterwarnings("ignore")


# In[4]:


#load data set
data = pd.read_csv(r'C:\Users\shiri\OneDrive\Desktop\pro Sci\new\adult_csv.csv', na_values='?')

display(Markdown("#### Shape of Data"))
print(data.shape)
display(Markdown("#### Column names"))
print(data.columns)
data.head()
#checking missing values
data.isnull().sum()
#drop the missing values
data = data.dropna()


# In[5]:


#data preparation
data.shape


# In[6]:


data.head().T


# In[7]:


data.columns


# In[8]:


data['class'].value_counts()


# In[9]:


Y_columns = ['sex', 'race']
ignore_columns = ['class']
cat_columns = []
num_columns = []

for col in data.columns.values:
    if col in Y_columns+ignore_columns:
        continue
    elif data[col].dtypes == 'int64':
        num_columns += [col]
    else:
        cat_columns += [col]


# In[10]:


#an overview of the numerical columns
print(data[num_columns].describe())


# In[11]:


#Frequency distribution of values in variables
for column in cat_columns:
    print(column) 
    print(data[column].value_counts())


# In[38]:


#funtions are taken from the following sources
#inspried by the kaggle tutorial https://www.kaggle.com/code/nathanlauga/ethics-and-ai-how-to-prevent-bias-on-ml/notebook and
#the github https://github.com/BiancaZimmer/Stat-ML-Fairness/blob/d1b04f8013475436ac8cc0e8f8852b2351c14a68/representativenessfairness.py


# In[12]:


def target_distribution(y_var, data):
    val = data[y_var]

    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({'font.size': 13})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    cnt = val.value_counts().sort_values(ascending=True)
    labels = cnt.index.values

    sizes = cnt.values
    colors = sns.color_palette("PuBu", len(labels))

    #------------COUNT-----------------------
    ax1.barh(cnt.index.values, cnt.values, color=colors)
    ax1.set_title('Count plot of '+y_var)

    #------------PERCENTAGE-------------------
    ax2.pie(sizes, labels=labels, colors=colors,autopct='%1.0f%%', shadow=True, startangle=130)
    ax2.axis('equal')
    ax2.set_title('Distribution of '+y_var)
    plt.show()


# In[13]:


var = 'sex'
target_distribution(y_var=var, data=data)


# In[ ]:


var = 'sex'
target_distribution(y_var=var, data=data)


# In[14]:


var = 'race'
target_distribution(y_var=var, data=data)


# In[15]:


var = 'class'
target_distribution(y_var=var, data=data)


# In[16]:


var = 'education'
target_distribution(y_var=var, data=data)


# In[39]:


data['Frequency'] = 1
freq_target = data[['sex', 'race', 'Frequency']]
del data['Frequency']
freq_target = freq_target.groupby(by=['sex', 'race']).count() / len(data)
print(freq_target.sort_values(by='Frequency', ascending=False))


# In[45]:


plot_histo(data, col='class',Y_columns = Y_columns)


# In[40]:


#If we based our model on the most frequents values we found that by default there is 59% of chance that the income class is a white man.


# In[41]:


plot_bar(data, col='occupation',Y_columns=Y_columns)


# In[42]:


plot_bar(data, col='education',Y_columns=Y_columns)


# In[43]:


plot_bar(data, col='relationship', Y_columns=Y_columns)


# In[44]:


display(Markdown("#### Number of sex = 1 "))
print(data[data['sex'] ==  'Male'].shape[0])
print(pd.crosstab(data['class'], data['sex']== "Male", normalize='index'))


# In[35]:


display(Markdown("#### Number of Sex = Female "))
print(data[data['sex'] == 'Female'].shape[0])
print(pd.crosstab(data['class'], data['sex']=="Female", normalize='index'))


# In[ ]:




