import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("bank.csv")

# Features and Target
X = df[["Age", "Salary"]]
y = df["Subscribed"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Graph
plt.bar(["Accuracy"], [accuracy])
plt.title("Decision Tree Accuracy")
plt.savefig("output.png")
plt.show()