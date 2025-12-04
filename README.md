# car-price-prediction
This is a Car Price Prediction Regression project. We are using multiple models: Linear Regression,Lasso,Ridge,Decision Tree,Random Forest,and KNN


from google.colab import drive
drive.mount('/content/drive')

import pandas as pd


df=pd.read_csv("car_price.csv")

df.head()

df.isnull().sum()

df.info()

#Car_Names

from sklearn.preprocessing import LabelEncoder
Car_Name_le=LabelEncoder()
df["car_name"]=Car_Name_le.fit_transform(df["car_name"])

#Fuel_type

from sklearn.preprocessing import LabelEncoder
Fuel_Type_le=LabelEncoder()
df["fuel_type"]=Fuel_Type_le.fit_transform(df["fuel_type"])


#seller type

Seller_Type_le=LabelEncoder()
df["seller_type"]=Seller_Type_le.fit_transform(df["seller_type"])


#Transmission

Transmission_le=LabelEncoder()
df["transmission"]=Transmission_le.fit_transform(df["transmission"])


df.shape

input_data=df.iloc[:,:-1]
output_data=df["selling_price"]

org_data=input_data.columns



from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaled_data_array=scaler.fit_transform(input_data)
input_data=pd.DataFrame(scaled_data_array,columns=org_data)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(input_data,output_data,test_size=0.2,random_state=42)

from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

sns.heatmap(df.corr(),annot=True)
plt.show()

lr=LinearRegression()
lr.fit(x_train,y_train)
lr.score(x_test,y_test)*100,lr.score(x_test,y_test)*100

lr1=Lasso(alpha=0.05)
lr1.fit(x_train,y_train)
lr1.score(x_train,y_train)*100,lr1.score(x_test,y_test)*100

lr2=Ridge(alpha=0.05)
lr2.fit(x_train,y_train)
lr2.score(x_train,y_train)*100,lr2.score(x_test,y_test)*100

dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
dt.score(x_train,y_train)*100,dt.score(x_test,y_test)*100

rf=RandomForestRegressor()
rf.fit(x_train,y_train)
rf.score(x_train,y_train)*100,rf.score(x_test,y_test)*100

knn=KNeighborsRegressor()
knn.fit(x_train,y_train)
knn.score(x_train,y_train)*100,knn.score(x_test,y_test)*100



lr.predict(x_test)

