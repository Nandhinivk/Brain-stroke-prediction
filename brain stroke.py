import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import pickle

# Load dataset
df = pd.read_csv("stroke_data.csv")

# Exploratory Data Analysis (EDA)
print(df.head())
print(df.info())
print(df.describe())

# Visualizing data
df['stroke'].value_counts().plot(kind='bar', title='Stroke Distribution')
plt.show()
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Preprocessing
X = df.drop(columns=['stroke'])  # Features
y = df['stroke']  # Target
X = pd.get_dummies(X)  # Convert categorical to numerical
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ANN Model
model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=10, validation_data=(X_test, y_test))

# Save model
model.save("stroke_model.h5")
pickle.dump(scaler, open("scaler.pkl", "wb"))

# Django Views
@csrf_exempt
def predict(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))['features']
        data = np.array(data).reshape(1, -1)
        data = scaler.transform(data)
        prediction = model.predict(data)[0][0]
        result = "High Risk" if prediction > 0.5 else "Low Risk"
        return JsonResponse({"prediction": result})

@csrf_exempt
def chatbot(request):
    if request.method == 'POST':
        user_input = json.loads(request.body.decode('utf-8'))['message']
        responses = {
            "What is stroke?": "A stroke occurs when the blood supply to part of the brain is interrupted or reduced.",
            "How to prevent stroke?": "Maintain a healthy diet, exercise regularly, and manage blood pressure."
        }
        return JsonResponse({"response": responses.get(user_input, "Sorry, I don't understand.")})
