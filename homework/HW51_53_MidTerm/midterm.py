
# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train_data.csv')
dataset=dataset.fillna(0)   #Nan補0

dataset['to_poi_ratio'] = dataset['from_this_person_to_poi'] / dataset['from_messages']
dataset['shared_poi_ratio'] = dataset['shared_receipt_with_poi'] / dataset['to_messages']


#dataset=dataset.fillna(0)   #Nan補0

X = dataset[['bonus', 'total_stock_value', 'expenses', 'exercised_stock_options', 'to_poi_ratio','shared_poi_ratio']].values
y = dataset["poi"].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.0, random_state = 7524)

#Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)

# Fitting Random Forest Classification to the Training set
import xgboost as xgb
xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=7524)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(xgb_model, X_train, y_train, cv=5)
print("Average score of training set : ")
print(scores.mean())

print("Average precision of training set : ")
precision=cross_val_score(xgb_model, X_train, y_train, cv=5,scoring='precision').mean()
print(precision)

print("Average recall of training set : ")
recall=cross_val_score(xgb_model, X_train, y_train, cv=5,scoring='recall').mean()
print(recall)

print("Average F1 score of training set : ")
F1score=2*precision*recall/(precision+recall)
print(F1score)

xgb_model.fit(X_train,y_train)
#
##Predict Test data File
dataset = pd.read_csv('test_features.csv')
dataset=dataset.fillna(0)   #Nan補0
dataset['to_poi_ratio'] = dataset['from_this_person_to_poi'] / dataset['from_messages']
dataset['shared_poi_ratio'] = dataset['shared_receipt_with_poi'] / dataset['to_messages']
#dataset=dataset.fillna(0)   #Nan補0


X_test = dataset[['bonus', 'total_stock_value', 'expenses', 'exercised_stock_options', 'to_poi_ratio','shared_poi_ratio']].values

#X_test = sc.transform(X_test)

y_pred = xgb_model.predict_proba(X_test)

result= (pd.DataFrame(data=[dataset["name"],y_pred[:,1]],index=['name','poi'])).T
i=1
result.to_csv('Prediction.csv',index=False)

