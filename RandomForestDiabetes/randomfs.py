import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Read data into pandas dataframe
df = pd.read_csv('../data/diabetes.csv')

# Define Feature Matrix (X) and Label Array (y)
X = df.drop(['Outcome'], axis=1)
y = df['Outcome']

# Define Feature Matrix (X) and Label Array (y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=0)

lr= RandomForestClassifier(n_estimators=53, n_jobs=1, random_state=8)
lr.fit(X,y)


# Serialize the model and save
import joblib
joblib.dump(lr, 'randomfs.pkl')
print("Random Forest Model Saved")

# Load the model
lr = joblib.load('randomfs.pkl')

# Save features from training
rnd_columns = list(X_train.columns)
joblib.dump(rnd_columns, 'rnd_columns.pkl')
print("Random Forest Model Colums Saved")