import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

dataset = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")

dataset.head()
dataset.info()  # Age and cabin has missing data
dataset.drop('Cabin', axis=1, inplace=True)
dataset.fillna(value=dataset.Age.mean(), inplace=True)
dataset.drop(dataset.loc[dataset.Embarked.map(type) != str].index, inplace=True)  # Deleting noise
dataset["Relatives"] = dataset.Parch + dataset.SibSp
dataset["AgeClass"] = dataset.Age * dataset.Pclass
plt.bar(dataset.Embarked, dataset.Survived)
plt.show()  # Pretty evenly spreaded
# dataset.loc[(dataset.Survived == 1) & (dataset.Sex == 'male')].count()[0] / dataset.loc[dataset.Sex == 'male'].count()[
#     0]  # Percentage of survived men: .189
# dataset.loc[(dataset.Survived == 1) & (dataset.Sex == 'female')].count()[0] / \
# dataset.loc[dataset.Sex == 'female'].count()[
#     0]  # Percentage of survived women: .742
numerical_features = ["Relatives",  "Fare", "AgeClass"]
categorical_features = ["Sex", "Embarked"]
X_numerical = dataset[numerical_features].values
Y = dataset["Survived"].values
X_categorical = dataset[categorical_features].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_numerical)
encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(X_categorical)
X_train_encoded = encoder.transform(X_categorical).toarray()
X = np.hstack((X_train_scaled, X_train_encoded))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.20, random_state=42)
# X_train = np.hstack((X_train_scaled, X_categorical))
# X_train[np.where(X_train == 'male')] = 1
# X_train[np.where(X_train == 'female')] = 0
play = pd.DataFrame(np.hstack((X, Y.reshape(889, 1))),
                    columns=numerical_features + encoder.categories_[0].tolist() + encoder.categories_[1].tolist() + [
                        "Survived"], index=dataset.index)
corr_matrix = play.corr()
corr_matrix["Survived"].sort_values(ascending=False)
# ****Logistic Regression****
param_grid = {
    "learning_rate": ["constant", "optimal"],
    "alpha": [1e-5, 1e-4, 1e-3],
    "eta0": [0.01, 0.03, 0.1, 1]
}
regressor = SGDClassifier(loss='log', penalty='l2', fit_intercept=True, n_iter_no_change=5)
grid_search = GridSearchCV(regressor, param_grid, cv=3, scoring='roc_auc')
grid_search.fit(X_train, Y_train)
print(grid_search.best_params_)  # {'alpha': 0.001, 'eta0': 0.03, 'learning_rate': 'constant'}
regressor_best = grid_search.best_estimator_
pred = regressor_best.predict(X_test)
print("ROC AUC for SGD is {0:.3f}".format(roc_auc_score(Y_test, pred)))  # # .78

# ****Random Forest****
param_grid = {
    "max_depth": [15, 20, 25],
    "min_samples_split": [5, 10, 15],
    "max_features": ['auto', 'sqrt'],
    "min_samples_leaf": [5],
    "n_estimators": [250, 275, 300]
}
rf = RandomForestClassifier(n_jobs=-1)
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, Y_train)
print(
    grid_search.best_params_)  # {'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 5,
# 'min_samples_split': 10, 'n_estimators': 250}
rf_best = grid_search.best_estimator_
pred = rf_best.predict(X_test)
rf_best.feature_importances_
print("ROC AUC for Random Forest is {0:.3f}".format(roc_auc_score(Y_test, pred)))  # .78
play.columns
# ****SVMClassifier****
param_grid = [
    {
        'kernel': ['linear'],
        'C': [100, 300, 500]
    },
    {
        'kernel': ['rbf'],
        'gamma': [1e-3, 1e-4, "auto"],
        'C': [10, 100, 1000],
    }
]
svc = SVC()
grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train, Y_train)
print(grid_search.best_params_)  # {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}
svc_best = grid_search.best_estimator_
pred = svc_best.predict(X_test)
print("ROC AUC for SVC is {0:.3f}".format(roc_auc_score(Y_test, pred)))  # .79

# ****Naive Bayes****
nb = GaussianNB()
nb.fit(X_train, Y_train)
pred = nb.predict(X_test)
roc_auc_score(Y_test, pred)  # .786

# ****Predicting Kaggle dataset****
data_test.drop('Cabin', axis=1, inplace=True)
data_test.fillna(value=dataset.Age.mean(), inplace=True)
data_test["Relatives"] = data_test.Parch + data_test.SibSp
data_test["AgeClass"] = data_test.Age * data_test.Pclass
X_test_numerical = data_test[numerical_features].values
X_test_categorical = data_test[categorical_features].values
X_test_scaled = scaler.transform(X_test_numerical)
X_test_encoded = encoder.transform(X_test_categorical).toarray()
X_kaggle = np.hstack((X_test_scaled, X_test_encoded))
pred = pd.DataFrame(rf_best.predict(X_kaggle), index=data_test.PassengerId, columns=['Survived'])
pred.to_csv("pred.csv")
