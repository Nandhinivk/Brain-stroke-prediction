pip install django numpy pandas tensorflow scikit-learn seaborn matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Load Dataset
df = pd.read_csv('brain_stroke.csv')
# Data Preprocessing
df.dropna(inplace=True)
df = pd.get_dummies(df, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'], drop_first=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#Exploratory Data analysis
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x='stroke',data=df)
plt.title('Distribution of Stroke')
plt.show
#Train and test the data
x = df.drop(columns=['stroke'])
y = df['stroke']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# ANN Model
ann_model = Sequential([
    Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
ann_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann_model.fit(x_train, y_train, epochs=50, batch_size=10, validation_data=(x_test, y_test))
#Logistic regression
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("ANN Accuracy:", ann_model.evaluate(x_test, y_test)[1])
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
# Visualization
def plot_visualizations():
    plt.figure(figsize=(12,6))
    sns.countplot(x='gender_Male', hue='stroke', data=df)
    plt.title('Stroke Cases by Gender')
    plt.show()
    
plot_visualizations()
# Age Distribution Histogram
plt.figure(figsize=(12,6))
sns.histplot(df['age'], bins=20, kde=False, color='steelblue')
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age Distribution')
plt.show()
# Average Glucose Level Distribution Histogram
plt.figure(figsize=(12,6))
sns.histplot(df['avg_glucose_level'], bins=50, kde=False, color='steelblue')
plt.xlabel('Avg Glucose Level')
plt.ylabel('Count')
plt.title('Average Glucose Level Distribution')
plt.show()


import joblib
joblib.dump('scaler', 'scaler.pkl')