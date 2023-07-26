import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#Load the Data
df = pd.read_csv("iris.csv")

print(df.head())

x = df.drop(columns='species')
y = df.species

#split the data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

#Feature scaling
sc = StandardScaler()
xtrain = sc.fit_transform(x_train)
xtest = sc.transform((x_test))

#Instantiate the model
classifier = RandomForestClassifier()
classifier.fit(xtrain,y_train)

#Make pickle file
import pickle
pickle.dump(classifier, open("model.pkl", 'wb'))