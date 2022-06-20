# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the necessary packages using import statement.
2. Read the given csv file and print the number of contents to be displayed.
3. Split the dataset using train_test_split.
4. Calculate Y_Pred and accuracy.
5. Display the result.


## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: JAYATHRAA V
RegisterNumber:  212219220018

import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')
data.head()
data.info()
data.isnull().sum()
x=data["EmailText"].values
y=data["Label"].values
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

```

## Output:
Data.head():

![image](https://user-images.githubusercontent.com/107881970/174665713-f1f01afe-c5d4-4ab5-9d19-123a6a6fd679.png)


Data.info():

![image](https://user-images.githubusercontent.com/107881970/174665731-67243c0e-795a-4c53-8441-e2fca8952509.png)


Data.isnull().sum():

![image](https://user-images.githubusercontent.com/107881970/174665753-543b728e-3b9e-43bc-8bcc-7200f0c46508.png)


Y_Pred:

![image](https://user-images.githubusercontent.com/107881970/174665774-8cdfe07d-f7ef-4d4d-b061-262490346f07.png)

Accuracy:

![image](https://user-images.githubusercontent.com/107881970/174665796-d3a84654-f429-47e5-85c8-ef85b5c957bb.png)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
