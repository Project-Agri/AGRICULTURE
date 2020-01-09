
# coding: utf-8

# In[213]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[214]:


df = pd.read_excel("E:\\WATIN INDIA\\TASKS\\Agriculture\\Original dataset\\Agriculture_3_states.xlsx")
df.head()


# In[215]:


df.shape


# In[216]:


df.dtypes


# In[217]:


df['Production']=pd.to_numeric(df["Production"], errors ='coerce')
### Here it converted '=' to numeric


# In[218]:


df.dtypes


# In[220]:


df.isnull().sum()


# In[221]:


df = df.dropna(how='any',axis=0)


# In[222]:


df.isnull().sum()


# In[223]:


df.describe()


# In[224]:


print("Total : ", df[df.Production == 0].shape[0])
### In all 28824 records in Production there are 1067 zeros.


# In[225]:


df.groupby("State_Name").size()


# In[226]:


sns.countplot(df['State_Name'],label="Count")


# In[228]:


df.groupby("District_Name").size()


# In[229]:


sns.countplot(df['District_Name'],label="Count")


# In[230]:


df.groupby("Season").size()


# In[231]:


sns.countplot(df['Season'],label="Count")


# In[232]:


df.groupby("Crop").size()


# In[233]:


sns.countplot(df['Crop'],label="Count")


# In[234]:


plt.figure(figsize=(16, 6))
sns.boxplot(df["Area"])


# In[236]:


corr = df.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1)
corr


# In[237]:


f, ax = plt.subplots(figsize =(16, 10)) 
sns.heatmap(corr, ax = ax, cmap ="YlGnBu", linewidths = 0.1)


# ### MAX AND MIN PRODUCTION TOP 5

# In[238]:


max_production = df.sort_values(by='Production', ascending=False)
max_production.head()


# In[240]:


# State which has overall maximum production.
sns.barplot(x='State_Name',y='Production',data=max_production, color = 'yellow')
plt.show()


# In[244]:


# District which has overall maximum production
max_district = df.groupby(['District_Name'])['Production'].sum().sort_values(ascending = False)
max_district.head()


# In[245]:


# Maximum Area
max_area = df.sort_values(by='Area', ascending=False)
max_area.head()


# In[259]:


### which year got max production
max_Y = df.groupby(['Crop_Year'])['Production'].max().sort_values(ascending = False)
max_Y


# In[253]:


# Which season has got maximum production
max_season= df.groupby(['Season'])['Production'].max()
max_season
# Whole year has the maximum production


# In[256]:


# Crop which has got maximum production in 17 years
max_crop= df.groupby(['Crop'])['Production'].sum().sort_values(ascending  = False)
max_crop


# In[258]:


max_crop_per_year= df.groupby(['Crop','Crop_Year'])['Production'].sum().sort_values(ascending  = False)
max_crop_per_year


# In[203]:


import mysql
import mysql.connector


# In[204]:


mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="root",
  database="Agriculture"
)


# In[205]:


mycursor = mydb.cursor()


# In[185]:


sql = "select * from agr         inner join         (select crop, count(crop) as max1 from         agr         group by crop         order by max1 desc         limit 30) T         on         agr.crop = T.crop and agr.area and agr.production;";


# In[186]:


mycursor.execute(sql)

myresult = mycursor.fetchall()

for x in myresult:
  print(x)


# In[187]:


data = pd.DataFrame(myresult)
data.head()


# In[188]:


data.shape


# In[189]:


data = data.iloc[:,0:7]
data.head()


# In[190]:


data.columns =['State_Name', 'District_Name','Crop_Year','Season','Crop','Area','Production'] 


# In[191]:


data.head()


# In[192]:


dummy = pd.get_dummies(data['State_Name'])
dummy.head()                          


# In[193]:


dummy_S = pd.get_dummies(data['Season'])
dummy_S.head()                          


# In[194]:


data = pd.concat([data,dummy], axis = 1)
data.head()


# In[195]:


data = pd.concat([data,dummy_S], axis = 1)
data.head()


# In[196]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()


# In[197]:


data['District_Name_L'] = labelencoder.fit_transform(data['District_Name'])
data['Crop_L'] = labelencoder.fit_transform(data['Crop'])


# In[198]:


data.head()


# In[199]:


X = data.iloc[:,[2,5,7,8,9,10,11,12,13,14]]
X.head()


# In[200]:


Y = data.iloc[:,6]
Y.head()


# In[201]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=7)


# In[164]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
model = lr.fit(X_train, y_train)


# In[165]:


y_pred = lr.predict(X_test)


# In[166]:


df_lr = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})


# In[167]:


df_lr.head()


# In[172]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
rf.fit(X_train, y_train)


# In[173]:


predictions = rf.predict(X_test)


# In[174]:


df_R = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})


# In[175]:


df_R.head()

