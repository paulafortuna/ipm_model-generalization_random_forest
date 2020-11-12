

## Import the random forest model.
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import r2_score
from rfpimp import permutation_importances
from sklearn.model_selection import cross_val_score
import numpy as np

data = pd.read_csv("./2020-10-21_dataset_ready_to_analyse.csv")

y_train = (data['F1_macro'] >= 0.7).astype('int64')
X_train = data.drop(columns=['F1_macro'])




## This line instantiates the model.
rf = RandomForestClassifier(n_estimators=50)

## Fit the model on your training data.
rf.fit(X_train, y_train)


# 10-Fold Cross validation
print("metric")
print(np.mean(cross_val_score(rf, X_train, y_train, cv=5, scoring='f1_macro')))

#feature importance
def r2(rf, X_train, y_train):
    return r2_score(y_train, rf.predict(X_train))


perm_imp_rfpimp = permutation_importances(rf, X_train, y_train, r2)


print(perm_imp_rfpimp)

perm_imp_rfpimp['features'] = perm_imp_rfpimp.index