import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# Reading from csv(comma separated values)

dataset = pd.read_csv("7.1 Social_Network_Ads.csv")
X = dataset.iloc[:, 2:-1].values
Y = dataset.iloc[:, -1].values

# Dividing dataset into separate training and testing dataset

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.25, random_state=5)

# Scaling the values so one feature doesn't have more effect on model as the other eg. salary is numerically higher
# than age but doesn't necessarily has a greater effect

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Making the model now

classifier = KNeighborsClassifier(n_neighbors=5, metric="minkowski")
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

# calculating accuracy of the model

confusionMatrix = confusion_matrix(Y_test, Y_pred)

# confusion matrix index 0, 0 => Value should be 0, is 0
# confusion matrix index 0,1 => Value should be 0, is 1 and so on..
# Accuracy = Right Guess / Total = ((0,0)+(1,1)) / ((0,0)+(1,1)+(0,1)+(1,0))
# For this model accuracy  = .92 or 92%
