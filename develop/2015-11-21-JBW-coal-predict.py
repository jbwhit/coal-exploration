
# coding: utf-8

# # Make a prediction about the coal production

# In[3]:

get_ipython().magic(u'matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set();


# In[4]:

df = pd.read_csv("../data/cleaned_coalpublic2013.csv", index_col='MSHA ID')
df.head()


# In[5]:

len(df)


# In[6]:

for column in df.columns:
    print column


# In[7]:

df.log_production.hist()


# In[8]:

df.Mine_Status.unique()


# In[9]:

df[['Mine_Status', 'log_production']].groupby('Mine_Status').mean()


# # Predict the Production of coal mines

# In[10]:

for column in df.columns:
    print column


# In[11]:

df.Year.unique()


# In[12]:

features = ['Average_Employees',
            'Labor_Hours',
           ]

categoricals = ['Mine_State',
                'Mine_County',
                'Mine_Status',
                'Mine_Type',
                'Company_Type',
                'Operation_Type',
                'Union_Code',
                'Coal_Supply_Region',
               ]

target = 'log_production'


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[14]:

fig = plt.subplots(figsize=(14,8))
sns.set_context('poster')
sns.violinplot(y="Mine_Status", x="log_production", data=df,
               split=True, inner="stick", )
plt.tight_layout()


# In[15]:

fig = plt.subplots(figsize=(14,8))
sns.set_context('poster')
sns.violinplot(y="Company_Type", x="log_production", data=df, 
               split=True, inner="stick");
plt.tight_layout()


# In[16]:

df['Company_Type'].unique()


# In[18]:

pd.get_dummies(df['Company_Type']).sample(50).head()


# In[19]:

dummy_categoricals = []
for categorical in categoricals:
    print categorical, len(df[categorical].unique())
    # Avoid the dummy variable trap!
    drop_var = sorted(df[categorical].unique())[-1]
    temp_df = pd.get_dummies(df[categorical], prefix=categorical)
    df = pd.concat([df, temp_df], axis=1)
    temp_df.drop('_'.join([categorical, str(drop_var)]), axis=1, inplace=True)
    dummy_categoricals += temp_df.columns.tolist()


# In[20]:

dummy_categoricals[:10]


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# # Build our model

# In[21]:

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor


# In[22]:

len(dummy_categoricals)


# In[23]:

train, test = train_test_split(df, test_size=0.3)


# In[ ]:




# In[25]:

rf = RandomForestRegressor(n_estimators=100, oob_score=True)


# In[27]:

rf.fit(train[features + dummy_categoricals], train[target])


# In[29]:

fig = plt.subplots(figsize=(8,8))
sns.regplot(test[target], rf.predict(test[features + dummy_categoricals]))
plt.ylabel("Predicted production")
plt.xlim(0, 22)
plt.ylim(0, 22)
plt.tight_layout()


# In[30]:

from sklearn.metrics import explained_variance_score, r2_score, mean_squared_error


# In[31]:

predicted = rf.predict(test[features + dummy_categoricals])
r2_score(test[target], predicted)


# In[32]:

explained_variance_score(test[target], predicted)


# In[33]:

mean_squared_error(test[target], predicted)


# In[34]:

rf_importances = pd.DataFrame({'name':train[features + dummy_categoricals].columns,
                               'importance':rf.feature_importances_
                              }).sort_values(by='importance', 
                                              ascending=False).reset_index(drop=True)
rf_importances.head(20)


# In[ ]:



