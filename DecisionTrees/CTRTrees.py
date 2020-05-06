import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

n_rows = 300000
train_ds = pd.read_csv('./avazu-ctr-prediction/train.csv', nrows=n_rows)
train_ds.info()

Y = train_ds['click'].values
X = train_ds.drop(['click', 'id', 'site_id', 'app_id', 'device_id', 'hour'], axis=1).values

n_train = int(n_rows * 0.9)  # not done randomly because data is ordered chronologically
X_train = X[:n_train]
Y_train = Y[:n_train]
X_test = X[n_train:]
Y_test = Y[n_train:]

# Transforming categorical to numerical data as trees cannot accept categorical features
enc = OneHotEncoder(handle_unknown='ignore')  # Won't give an error if it sees a new feature after fit
X_train_enc = enc.fit_transform(X_train)
X_test_enc = enc.transform(X_test)

parameters = {'max_depth': [3, 10, None]}
decistion_tree = DecisionTreeClassifier(criterion='gini', min_samples_split=30)
grid_search = GridSearchCV(decistion_tree, parameters, n_jobs=-1, cv=3, scoring='roc_auc')
grid_search.fit(X_train_enc, Y_train)

# Results from cross validation
print(grid_search.best_params_)
decision_tree_best = grid_search.best_estimator_
pos_prob = decision_tree_best.predict_proba(X_test_enc)[:, 1]
print('The ROC AUC on testing set is: {0:.3f}'.format(roc_auc_score(Y_test, pos_prob)))  # Score is .717

# Using Random Forest
random_forest = RandomForestClassifier(n_estimators=100, criterion='gini', min_samples_split=30, n_jobs=-1)
grid_search = GridSearchCV(random_forest, parameters, n_jobs=-1, cv=3, scoring='roc_auc')
grid_search.fit(X_train_enc, Y_train)
print(grid_search.best_params_)

# Results from cross validation
random_forest_best = grid_search.best_estimator_
pos_prob = random_forest_best.predict_proba(X_test_enc)[:, 1]
print('The ROC AUC on testing set is: {:0.3f}'.format(roc_auc_score(Y_test, pos_prob)))  # Score is .757, better than before
