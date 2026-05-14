# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset into a DataFrame and explore its contents to understand the data structure.
2.Separate the dataset into independent (X) and dependent (Y) variables, and split them into training and testing sets.
3.Create a linear regression model and fit it using the training data.
4.Predict the results for the testing set and plot the training and testing sets with fitted lines.
5.Calculate error metrics (MSE, MAE, RMSE) to evaluate the model’s performance.

## Program:
Program to implement the simple linear regression model for predicting the marks scored.
**<br>Developed by: Sridhar c <br>
RegisterNumber: 212225040425**
~~~py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error ,mean_absolute_error,r2_score
data={"hours":[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
      "marks":[35, 40, 50, 55, 60, 65, 70, 75, 80, 85]
     }
df=pd.DataFrame(data)
x=df[["hours"]]
y=df["marks"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
y_predict=model.predict(x_test)
mse=mean_squared_error(y_test,y_predict)
mae=mean_absolute_error(y_test,y_predict)
r2=r2_score(y_test,y_predict)
print(f"The predicted value is {y_predict}")
print(f"The mean squared error value is {mse}")
print(f"The mean  absolute error value is {mae}")
print(f"The r2 score value is {r2}")
plt.scatter(x,y,color="green",label="Actual line")
plt.plot(x,model.predict(df[["hours"]]),color="red",label="fixed line")
plt.title("hour vs mark",fontsize=19,color="blue")
plt.xlabel("hours",fontsize=15)
plt.ylabel("marks",fontsize=15)
plt.legend()
plt.show()
hour=float(input("Enter the hours"))
new_predicted_data=model.predict([[hour]])
print(f"marks for studying {hour} hours {new_predicted_data[0]:.2f}")
~~~
## Output:
<img width="873" height="584" alt="Screenshot from 2026-05-14 09-17-00" src="https://github.com/user-attachments/assets/d7cf0dc6-9778-437a-a877-61992a7e6089" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
