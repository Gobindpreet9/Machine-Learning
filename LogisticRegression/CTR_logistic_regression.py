# Using logistic regression with SGD to make a very scalable classifying model
import timeit
import pandas as pd
import pyspark
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import OneHotEncoder

n_rows = 100000 * 11
df = pd.read_csv("../DecisionTrees/avazu-ctr-prediction/train.csv", nrows=n_rows)

y = df['click'].values
x = df.drop(['click', 'id', 'site_id', 'app_id', 'device_id', 'hour'], axis=1).values

n_train = 100000 * 10
X_train = x[:n_train]
Y_train = y[:n_train]
X_test = x[n_train:]
Y_test = y[n_train:]

enc = OneHotEncoder(handle_unknown='ignore')  # Won't give an error if it sees a new feature after fit
enc.fit(X_train)
sgd_lr_online = SGDClassifier(loss='log', penalty='l2', fit_intercept=True, n_iter_no_change=5, learning_rate='optimal')

start_time = timeit.default_timer()
for i in range(10):
    x_train = X_train[i * 100000: (i + 1) * 100000]
    y_train = Y_train[i * 100000: (i + 1) * 100000]
    x_train_enc = enc.transform(x_train)
    sgd_lr_online.partial_fit(x_train_enc, y_train, classes=[0, 1])

print("*** %0.3f seconds ***" % (timeit.default_timer() - start_time))  # 8.42 seconds!!

x_test_enc = enc.transform(X_test)
pred = sgd_lr_online.predict_proba(x_test_enc)[:, 1]
print('Training samples: {0}, AUC on testing set: {1: .3f}'.format(n_train * 10, roc_auc_score(Y_test, pred)))
# .744 with l1 and .752 with l2
