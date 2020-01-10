
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


df = pd.read_excel("E:\\WATIN INDIA\\TASKS\\Agriculture\\Original dataset\\Agriculture_3_states.xlsx")
df.head()


# In[3]:


df.shape


# In[4]:


df.dtypes


# In[5]:


df['Production']=pd.to_numeric(df["Production"], errors ='coerce')
### Here it converted '=' to numeric


# In[6]:


df.dtypes


# In[7]:


df.isnull().sum()


# In[8]:


df = df.dropna(how='any',axis=0)


# In[9]:


df.isnull().sum()


# In[10]:


df.describe()


# In[11]:


print("Total : ", df[df.Production == 0].shape[0])
### In all 28824 records in Production there are 1067 zeros.


# In[24]:


df['CPI']=df['Production']/df['Area']
df.head()


# In[25]:


df.groupby("State_Name").size()


# In[26]:


sns.countplot(df['State_Name'],label="Count")


# In[27]:


df.groupby("District_Name").size()


# In[28]:


sns.countplot(df['District_Name'],label="Count")


# In[29]:


df.groupby("Season").size()


# In[30]:


sns.countplot(df['Season'],label="Count")


# In[31]:


df.groupby("Crop").size()


# In[32]:


sns.countplot(df['Crop'],label="Count")


# In[33]:


plt.figure(figsize=(16, 6))
sns.boxplot(df["Area"])


# In[34]:


corr = df.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1)
corr


# In[35]:


f, ax = plt.subplots(figsize =(16, 10)) 
sns.heatmap(corr, ax = ax, cmap ="YlGnBu", linewidths = 0.1)


# ### MAX PRODUCTION TOP 5

# In[36]:


max_production = df.sort_values(by='Production', ascending=False)
max_production.head()


# In[37]:


# State which has overall maximum production.
sns.barplot(x='State_Name',y='Production',data=max_production, color = 'yellow')
plt.show()


# In[38]:


# District which has overall maximum production
max_district = df.groupby(['District_Name'])['Production'].sum().sort_values(ascending = False)
max_district.head()


# In[39]:


# Maximum Area
max_area = df.sort_values(by='Area', ascending=False)
max_area.head()


# In[40]:


### which year got max production
max_Y = df.groupby(['Crop_Year'])['Production'].max().sort_values(ascending = False)
max_Y


# In[41]:


# Which season has got maximum production
max_season= df.groupby(['Season'])['Production'].max()
max_season
# Whole year has the maximum production


# In[42]:


# Crop which has got maximum production in 17 years
max_crop= df.groupby(['Crop'])['Production'].sum().sort_values(ascending  = False)
max_crop


# In[43]:


max_crop_per_year= df.groupby(['Crop','Crop_Year'])['Production'].sum().sort_values(ascending  = False)
max_crop_per_year


# In[44]:


import mysql
import mysql.connector


# In[45]:


mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="root",
  database="Agriculture"
)


# In[46]:


mycursor = mydb.cursor()


# In[83]:


sql = "select * from agr         inner join         (select crop, count(crop) as max1 from         agr         group by crop         order by max1 desc         limit 40) T         on         agr.crop = T.crop and agr.area and agr.production;";


# In[84]:


mycursor.execute(sql)

myresult = mycursor.fetchall()

for x in myresult:
  print(x)


# In[85]:


data = pd.DataFrame(myresult)
data.head()


# In[86]:


data.shape


# In[87]:


data = data.iloc[:,0:7]
data.head()


# In[88]:


data.columns =['State_Name', 'District_Name','Crop_Year','Season','Crop','Area','Production'] 


# In[89]:


data.head()


# In[90]:


data['CPI'] = data['Production']/data['Area']
data.head()


# In[91]:


dummy = pd.get_dummies(data['State_Name'])
dummy.head()                          


# In[92]:


dummy_S = pd.get_dummies(data['Season'])
dummy_S.head()                          


# In[93]:


data = pd.concat([data,dummy], axis = 1)
data.head()


# In[94]:


data = pd.concat([data,dummy_S], axis = 1)
data.head()


# In[95]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()


# In[96]:


data['District_Name_L'] = labelencoder.fit_transform(data['District_Name'])
data['Crop_L'] = labelencoder.fit_transform(data['Crop'])


# In[97]:


data.head()


# In[98]:


X = data.iloc[:,[2,8,9,10,11,12,13,14,15]]
X.head()


# In[99]:


Y = data.iloc[:,7]
Y.head()


# In[100]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=7)


# In[101]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
model = lr.fit(X_train, y_train)


# In[112]:


print('Coefficients: \n', lr.coef_)


# In[113]:


print('Variance score: {}'.format(lr.score(X_test, y_test)))


# In[102]:


y_pred = lr.predict(X_test)


# In[103]:


df_lr = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})


# In[104]:


df_lr.head()


# In[105]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
rf.fit(X_train, y_train)


# ### SAVING INTO PICKLE FORMAT

# In[106]:


import pickle 
# Save the trained model as a pickle string. 
saved_model = pickle.dumps(rf)


# In[107]:


rf_from_pickle = pickle.loads(saved_model)


# In[108]:


predictions = rf_from_pickle.predict(X_test)


# In[109]:


df_rf = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})


# In[110]:


df_rf.head()


# In[115]:


from sklearn.metrics import r2_score
print(r2_score(y_test,predictions))

