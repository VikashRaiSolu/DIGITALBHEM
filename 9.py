import numpy as np 
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split  
from sklearn import metrics
from sklearn.datasets import load_iris


iris = load_iris()


df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target


X = df.iloc[:, :-1]  
y = df.iloc[:, -1]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)


classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)


print("\n-------------------------------------------------------------------------")
print('%-25s %-25s %-25s' % ('Original Label', 'Predicted Label', 'Correct/Wrong'))
print("-------------------------------------------------------------------------")
for i, label in enumerate(y_test):
    print('%-25s %-25s' % (label, y_pred[i]), end="")
    if label == y_pred[i]:
        print('%-25s' % ('Correct'))
    else:
        print('%-25s' % ('Wrong'))
print("-------------------------------------------------------------------------")
 

print("\nConfusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred))
print("-------------------------------------------------------------------------")
print("\nClassification Report:\n", metrics.classification_report(y_test, y_pred))
print("-------------------------------------------------------------------------")
print('Accuracy of the classifier is %0.2f' % metrics.accuracy_score(y_test, y_pred))
print("-------------------------------------------------------------------------")
