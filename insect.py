
#importing libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#loading data
data = pd.read_csv('C:/Datasets/insect.csv')
feature_cols = ['length', 'Width', 'weight']

#defining the target column
target_cols = 'species'

#extracting features and target from the data
X = data[feature_cols] # Features
y = data[target_cols] # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1,shuffle=True)

#create the model using the decision tree classifier
model = DecisionTreeClassifier()

#train the model using
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#calculating accuracy
accuracy = accuracy_score(y_test, y_pred)

#plotting graph
plt.plot(y_test, y_pred, 'o')
plt.title('Insect Decision Tree Classifier')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

#printing accuracy
print('Accuracy:', accuracy)


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

