
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB


# In[3]:


train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")


# In[4]:


train_data.head()


# In[5]:


train_data.info()


# In[6]:


train_data.describe()


# In[7]:


train_data.hist(figsize=(20,12))


# In[8]:


sns.barplot(y="Survived",x="Sex",data=train_data)


# In[9]:


female=train_data.groupby(["Sex"]).Survived.sum()[0]
print("Percentage of male suvived are : {:0.3f}%".format(100*female/train_data.groupby(["Sex"]).Survived.count()[0]))
male=train_data.groupby(["Sex"]).Survived.sum()[1]
print("Percentage of male suvived are : {:0.3f}%".format(100*male/train_data.groupby(["Sex"]).Survived.count()[1]))


# In[10]:


sns.barplot(y="Survived",x="Embarked",data=train_data)


# In[11]:


sns.barplot(y="Survived",x="SibSp",data=train_data)


# In[12]:


sns.barplot(y="Survived",x="Parch",data=train_data)


# In[13]:


p=sns.FacetGrid(col="Survived",data=train_data)
p.map(plt.hist,"Age")


# In[14]:


sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=train_data,
              palette={"male": "blue", "female": "pink"},
              markers=["*", "o"], linestyles=["-", "--"]);


# In[15]:


sns.pointplot(x="Pclass", y="Fare", hue="Survived", data=train_data,
              palette={0: "blue", 1: "pink"},
              markers=["*", "o"], linestyles=["-", "--"]);


# In[16]:


train_data["Fare_modified"]=np.ceil(train_data["Fare"] / 50)
sns.barplot(y="Survived",x="Fare_modified",data=train_data)


# ## Retain the original data
# ### Removing "modified_fare"

# In[17]:


train_data.drop("Fare_modified",axis=1,inplace=True)


# In[18]:


test_data["Survived"]=(test_data.Sex=="female").astype(int)
test_data.head()
test_data.shape


# In[19]:


test_data[["PassengerId","Survived"]].to_csv("data/predictions/female_live.csv",index=False)


# ## Accuracy on Kaggle: 0.76555
# ### Based on Gender only

# In[20]:


test_data.drop("Survived",axis=1,inplace=True)
test_data.shape


# ## Combining Test and Training Datasets of performing processing operations simulataneously
# ### Removing "Survived" column from training data and saving it for future

# In[21]:


survived_train=train_data["Survived"]


# In[22]:


train_data.drop("Survived",axis=1,inplace=True)


# In[23]:


data=pd.concat([train_data,test_data],sort=False)


# In[24]:


data.info()


# In[25]:


missing_cols=[i for i in data.columns if data[i].isnull().any()]
missing_cols


# In[26]:


data["Age"]=data.Age.fillna(data.Age.median())
data["Fare"]=data.Fare.fillna(data.Fare.median())


# In[27]:


missing_cols=[i for i in data.columns if data[i].isnull().any()]
missing_cols


# In[28]:


data=pd.get_dummies(data,columns=["Sex"],drop_first=True)


# In[29]:


data.head()


# In[39]:


cols_select=["Sex_male","Age","Fare","SibSp"]


# In[40]:


data[cols_select].head()


# In[41]:


data[cols_select].info()
data_new=data[cols_select]


# In[42]:


df_train=data_new.iloc[:891]
df_test=data_new.iloc[891:]


# ## As sklearn only uses numpy arrays so, changing dataframe to array

# In[53]:


X=df_train
test=df_test
#Remember to use the previously extacted column
y=survived_train.values


# ## Fitting models

# In[68]:


clf=DecisionTreeClassifier(max_depth=5)
# clf=RandomForestClassifier(max_depth=5,n_estimators=25)

clf


# In[69]:


clf.fit(X,y)
pred=clf.predict(test)
test_data["Survived"]=pred


# In[70]:


test_data[["PassengerId","Survived"]].to_csv("data/predictions/DT.csv",index=False)
# test_data[["PassengerId","Survived"]].to_csv("data/predictions/RF.csv",index=False)
# test_data[["PassengerId","Survived"]].to_csv("data/predictions/RF.csv",index=False)


# ## Accuracy on Kaggle: 0.77990  DecisionTreeClassifier(max_depth=5)
# ### Based on ""Sex_male","Age","Fare","SibSp"
