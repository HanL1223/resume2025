---
title: "Password Strength Validation"
description: "Function to support senior user to understand their password strength"
dateString: Jun 2023
draft: false
tags: ["Python", 'Xgboost','Sklearn']
showToc: False
weight: 206
cover:
    image: "projects/grad-project/cover2.jpeg"
--- 
### Github link
🔗 [Function URL](https://lesterwithhistreasure.de/avoidingscam)
🖥 [Code](https://github.com/HanL1223/fit5120_backend_Django)
### Credit
Members of team TP8 for subject FIT5120/22 

{{< typeit 
  tag=h3
  speed=50
  breakLines=false
  loop=true
>}}
"Frankly, my dear, I don't give a damn." Gone with the Wind (1939)
"I'm gonna make him an offer he can't refuse." The Godfather (1972)
"Toto, I've a feeling we're not in Kansas anymore." The Wizard of Oz (1939)
{{< /typeit >}}


### Skill invlove
**Python** **Pandas** **Machine Learning** 

## Description


## Key Takeaways
- Combining DS skill to a real project. Experience on merging model develpment with backend(Django) and frontend(React).
- Agile Management. Here I also exposed to the full agile software development cycle of a product
- Method to improve the models

## Where to improve
- Use full training set to increase accuracy and F1 perfromance, reducee False positive and Falues Negative
- Different method to preprocess the data

## Function Code
```python
# For Data manipulation
import pandas as pd
import numpy as np

# For Data Visulization
import matplotlib.pyplot as plt
import seaborn as sns

#For Modelling and evaluation
from sklearn.model_selection import train_test_split,StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,BaggingClassifier,StackingClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier as KNN  
#For text preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer

#For model parameter saving and loading
import pickle
```

```python
# Import dataset
df = pd.read_csv('data.csv',on_bad_lines='skip') #ignore badline from dataset
df.isna().sum()
df.duplicated().sum()
#only 1 missing value, and no duplicate found,will drop the bad record directly
df.dropna(inplace = True)

#Import text data for all the weak passwords from rock you leak
df2 = pd.read_csv('rockyou.txt',delimiter='\t',header = None, names = ['password'],encoding='ISO-8859-1')
df2.dropna(inplace = True)
df2.drop_duplicates(inplace = True)
df2['strength'] = 0

df_full = pd.concat([df,df2],ignore_index=True)

# Compute the value counts of the Gender column
value_counts = df_full['strength'].value_counts()

# Set the number of samples to be drawn from each group
n_samples = value_counts.min()

# Group the dataframe by Gender and sample n_samples from each group
sampled_df = df.groupby('strength').apply(lambda x: x.sample(n=n_samples)).reset_index(drop=True)

# Print the sampled dataframe
print(sampled_df)

X = sampled_df['password']
y = sampled_df['strength']
#tokenize password

vectorizer = TfidfVectorizer(analyzer = 'char')
X = vectorizer.fit_transform(X)

#Save the vectorizer for backend use
with open("vectorizer2.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
```


```python
#Data preprocessing
def train_val_test_split(X,y,ratio):
    X_train,X_,y_train,y_ = train_test_split(X,y,test_size=ratio,stratify=y,random_state=1)
    X_val,X_test,y_val,y_test = train_test_split(X_,y_,test_size=.5,stratify=y_,random_state=1)
    return X_train,X_val,X_test,y_train,y_val,y_test
X_train,X_val,X_test,y_train,y_val,y_test = train_val_test_split(X,y,ratio=.25)

#Model Selection with CV

models = []  # Empty list to store all the models

# Appending models into the list

models.append(("Random forest", RandomForestClassifier(random_state=1)))
models.append(("Bagging", BaggingClassifier(random_state=1)))
models.append(("Xgboost", XGBClassifier(random_state=1, eval_metric="logloss")))
models.append(("lgbm", lgb.LGBMClassifier(random_state=1)))
models.append(('KNN',KNN()))

results = []  # Empty list to store all model's CV scores
names = []  # Empty list to store name of the models
score = []

# loop through all models to get the mean cross validated score

print("\n" "Cross-Validation Performance:" "\n")
# Use F1 since for password evaluation I want to ensure to minimal both FP and FN rate
for name, model in models:
    kfold = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=1
    )  # Setting number of splits equal to 5
    cv_result = cross_val_score(
        estimator=model, X=X_train, y=y_train, scoring='f1_macro', cv=kfold
    )
    results.append(cv_result)
    names.append(name)
    print("{}: {}".format(name, cv_result.mean()))



#Form CV result, XGBoost generate around 98.4% marco f1 score

#Using Xgboost provided the best f1_marco result, thus we can fine tune it

# defining model - XGBoost Hyperparameter Tuning
model = XGBClassifier(random_state=1, eval_metric="logloss")

# Parameter grid to pass in RandomizedSearchCV
param_grid = {
    "n_estimators": np.arange(150, 300, 50),
    "learning_rate": [0.0001,0.001,0.01,0.0015],
    "gamma": [0, 3, 5,7],
    "subsample": [0.5, 0.9,0.2,0.35],
    'reg_alpha':[0,1],
    'reg_lambda':[0,1]
}
# Calling RandomizedSearchCV
randomized_cv = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=20,  # sample 20 setting
    scoring='f1_macro',
    cv=3,
    random_state=1,
    n_jobs=-1,
)

# Fitting parameters in RandomizedSearchCV
randomized_cv.fit(X_train, y_train)

print(
    "Best parameters are {} with CV score={}:".format(
        randomized_cv.best_params_, randomized_cv.best_score_
    )
)


best_model = XGBClassifier(
**randomized_cv.best_params_,eval_metric = 'logloss',random_state = 1
)# building model with best parameters

print("\n" "Training Performance:" "\n")
best_model.fit(X_train, y_train)
scores = metrics.f1_score(y_train, best_model.predict(X_train),average='macro')
print(" {}".format( scores))

print("\n" "Validation Performance:" "\n")
val_scores = metrics.f1_score(y_val, best_model.predict(X_val),average='macro')
print("{}".format(val_scores))
# We can go ahead test the model if both training and validation performance are as expected


y_pred = best_model.predict(X_test)
metrics.f1_score(y_test,y_pred,average='macro')


import pickle

# Save the model into the pickle file for backend
with open("xgb_model.pkl", "rb") as f:
    pickle.dump(best_model,f)
```