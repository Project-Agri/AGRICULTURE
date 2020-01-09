import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import mysql.connector
from sklearn.ensemble import RandomForestRegressor

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="root",
  database="agriculture"
)

mycursor = mydb.cursor()

sql = "select * from agriculture \
        inner join \
        (select crop, count(crop) as max1 from \
        agriculture \
        group by crop \
        order by max1 desc \
        limit 30) T \
        on \
        agriculture.crop = T.crop \
        and agriculture.district_name in ('Coimbatore','srikakulam','thanjavur','tiruppur','visakapatnam', \
                                          'west godavari','east godavari');";
                                          
mycursor.execute(sql)

myresult = mycursor.fetchall()

for x in myresult:
  print(x)                                         
req_dt = pd.DataFrame(myresult) 
req_dt
labelencoder = LabelEncoder()
req_dt['State_Name'] = labelencoder.fit_transform(req_dt.iloc[:,[0]])
req_dt['District_Name'] = labelencoder.fit_transform(req_dt[1])
req_dt['Season'] = labelencoder.fit_transform(req_dt[3])
req_dt['Crop'] = labelencoder.fit_transform(req_dt[4])
req_dt
req_dt['CPI']=req_dt[6]/req_dt[5]
xx=req_dt.iloc[:,[2,9,10,11,12]]
xx
yy=req_dt.iloc[:,[13]]
yy
x_train,x_test,y_train,y_test=train_test_split(xx,yy,test_size=0.25,random_state=0)
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
rf.fit(x_train,y_train)
predictions = rf.predict(x_test)


"""
regression=LinearRegression()
regression.fit(xx,yy)
predict=regression.predict(xx)

re=[1997,0,1,0,9]
regression.predict([re])

poly=PolynomialFeatures(degree=4)
x_poly=poly.fit_transform(xx)
poly.fit(x_poly,yy)

Regression2=LinearRegression()
Regression2.fit(x_poly,yy)
y_pre=Regression2.predict(x_poly)
y_pre
Regression2.predict(poly.fit_transform([re]))  

rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
rf.fit(xx,yy)
predictions = rf.predict(xx)
acc_rf=rf.score(predictions,yy)
"""