import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.applications import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D
from keras.utils import to_categorical
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
import csv

# from tensorflow.keras.layers import Dense
#  loading the data set
loading = pd.read_csv(r'D:\Study folder\6 semester\ML\project\code\IRIS.csv')
print(loading.head(10))

# separating the features and target
data = loading.values
X = data[:, 0:4]
Y = data[:, 4]
# split the data into test and train dataset

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4)  # divide into 120 rows and 30 for test_size
print('***********************************************************************************************')
# Model 1 Logistic Regression
print('Prediction through Logistic Regression')
model_LR = LogisticRegression()
model_LR.fit(X_train, Y_train, )

predict1 = model_LR.predict(X_test)
print(accuracy_score(Y_test, predict1) * 100)

print('****************************************************************************************************')
# support vector machine algorithm      #MODEL 2
# Model fitting applying svc on traning set and predicting the test set
print('Prediction through Vector Machine Algorithm')
model_svc = SVC()
model_svc.fit(X_train, Y_train)

predict2 = model_svc.predict(X_test)
print(accuracy_score(Y_test, predict2) * 100)

# printing the values
for i in range(len(predict2)):
    print(Y_test[i], predict2)

print('****************************************************************************************************')

# Model 3 Decision tree classifier
print('Prediction through Decision Tree classifier ')
model_DTC = DecisionTreeClassifier()
model_DTC.fit(X_train, Y_train)
predict2 = model_DTC.predict(X_test)
print(accuracy_score(Y_test, predict2) * 100)
print('***************************************************************************************************')

print('Classification report of test data using Logistic Regression')
print(classification_report(Y_test, predict1))
print('*******************************************************************************************************')

# loading the data set
iris = pd.read_csv(r'D:\Study folder\6 semester\ML\project\code\iris_2.csv')
print(iris.head(10))
# Create a figure with three subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
# Plot the data on the subplots
ax1.scatter(iris['Sepal_Area'], iris['species'])
ax2.scatter(iris['Petal_Area'], iris['species'])
ax3.scatter(iris['Petal_Length'], iris['species'])

# Set the title for each subplot
ax1.set_title('Sepal_Area')
ax2.set_title('Petal_Area')
ax3.set_title('Petal_Length')

# Set the y-axis label for each subplot
ax1.set_ylabel('Species')
ax2.set_ylabel('Species')
ax3.set_ylabel('Species')

# Show the plot
plt.show()
print('*****************************************************************************************************')
# Model 1 Logistic Regression
print("****************************************************************************************************")
print("Prediction through second data_set")

# Separate the features and target variable
K = iris.drop("species", axis=1)
M = iris["species"]
# Preprocess the data
store = OneHotEncoder()
K_encoded = store.fit_transform(K)
scaler = StandardScaler()
K_scaled = scaler.fit_transform(K_encoded.toarray())
# Add noise to the features
noise = np.random.normal(loc=0, scale=0.5, size=K_scaled.shape)
K_noisy = K_scaled + noise

# Split the data into training and testing sets
K_train, K_test, M_train, M_test = train_test_split(K_noisy, M, test_size=0.65)
print("************************************************************************************************")
print("Prediction through Logistic Regression")
# Apply logistic regression
model = LogisticRegression()
model.fit(K_train, M_train)
# Make predictions on the test data
M_pred = model.predict(K_test)
# Print various metrics
print("Accuracy:", accuracy_score(M_test, M_pred) * 100)
print("***********************************************************************************************")
print('Prediction through Decision Tree classifier')
model_DTC = DecisionTreeClassifier()
model_DTC.fit(K_train, M_train)
predicting2 = model_DTC.predict(K_test)
print(accuracy_score(M_test, predicting2) * 100)

print("************************************************************************************************")
print('Prediction through Vector Machine Algorithm')
model_svc = SVC()
model_svc.fit(K_train, M_train)
predicting3 = model_svc.predict(K_train)
print(accuracy_score(M_train, predicting3) * 100)

print("************************************************************************************************")
print("Implementation of Resnet50 model")
# Load the Iris dataset
iris = load_iris()
E = iris.data
r = iris.target
r = to_categorical(r)
E_train, E_test, r_train, r_test = train_test_split(E, r, test_size=0.2)

model = ResNet50(weights='imagenet')

model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(4,)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])
import pandas as pd
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(E_train, r_train, epochs=100, batch_size=8, validation_data=(E_test, r_test))
#mmm = pd.DataFrame(hist)
#mmm.to_csv('XYZ.csv')


with open('Resnet_1.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["loss", "accuracy", "val_loss", "val_accuracy"])
    for i in range(len(hist.history['loss'])):
        writer.writerow([hist.history['loss'][i], hist.history['accuracy'][i], hist.history['val_loss'][i], hist.history['val_accuracy'][i]])

score = model.evaluate(E_test, r_test)
print('Test loss:', score[0] * 100)
print('Test accuracy:', score[1] * 100)

# Plot the training loss and accuracy


plt.figure(figsize=(6, 4))
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.ylim(0, 1)
plt.legend(['Train_Resnet', 'Validation_Resnet'], loc='upper left')
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.ylim(0, 1)
plt.legend(['Train_Resnet', 'Validation_Resnet'], loc='upper left')
plt.show()

# Load the Iris dataset
print("*************************************************************************************************")
print("Implementation of 1st Dataset Through MLP")
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create the MLP model
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(4,)))
model.add(Dense(3, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=10, validation_data=(X_test, y_test), verbose=1)
# Plot loss and accuracy graphs
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

with open('MLP_1.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["loss", "accuracy", "val_loss", "val_accuracy"])
    for i in range(len(history.history['loss'])):
        writer.writerow([history.history['loss'][i], history.history['accuracy'][i], history.history['val_loss'][i], history.history['val_accuracy'][i]])
# Loss curve
ax[0].plot(history.history['loss'], label='train')
ax[0].plot(history.history['val_loss'], label='test')
ax[0].set_title('Model Loss')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
plt.ylim(0, 1)
ax[0].legend(['Train_MLP', 'Validation_MLP'], loc='upper left')

# Accuracy curve
ax[1].plot(history.history['accuracy'], label='train')
ax[1].plot(history.history['val_accuracy'], label='test')
ax[1].set_title('Model Accuracy')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy')
plt.ylim(0, 1)
ax[1].legend(['Train_MLP', 'Validation_MLP'], loc='upper left')

plt.show()
test_loss, test_acc = model.evaluate(X_test, y_test)

print('Test loss:', test_loss * 100)
print('Test accuracy:', test_acc * 100)

predictions = model.predict(X_test)

print("**********************************************************************************************")

print("Implementation through MLP using dataset_2")
# Load the data
df = pd.read_csv(r"D:\Study folder\6 semester\ML\project\code\iris_2.csv")
le = LabelEncoder()
df['Origin'] = le.fit_transform(df['Origin'])
df["Color"] = le.fit_transform(df["Color"])
# Convert the dataframe into a numpy array
data = df.values

N_train, N_test, M_train, M_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], test_size=0.5, random_state=42)

# Scale the numeric columns in the dataset
scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Create an MLP model with one hidden layer of 10 neurons
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000)

# Train the MLP model and store the result in the 'result' variable
result = mlp.fit(N_train, M_train)
#loss curve
# Test the MLP model
accuracy = mlp.score(N_test, M_test)
print(f"Accuracy: {accuracy * 100}")

# Calculate loss
loss = result.loss_curve_
val_loss = mlp.loss_

# Create a graph of accuracy
plt.figure(figsize=(6, 4))
plt.plot(result.score(N_train, M_train))
plt.plot(result.score(N_test, M_test))
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.ylim(0, 1)
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Create a graph of loss
plt.figure(figsize=(6, 4))
plt.plot(loss)
plt.plot(val_loss)
plt.title('Model Loss MLP')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.ylim(0, 1)
plt.legend(['Train_MLP', 'Validation_MLP'], loc='upper left')
plt.show()


print('Implementation of Resnet_50 from Custom Dataset: ')
# Load the Iris dataset
iris = load_iris()
E = iris.data
r = iris.target
r = to_categorical(r)
E_train, E_test, r_train, r_test = train_test_split(E, r, test_size=0.2)

model = ResNet50(weights='imagenet')

model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(4,)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])
import pandas as pd
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(E_train, r_train, epochs=100, batch_size=8, validation_data=(E_test, r_test))
#mmm = pd.DataFrame(hist)
#mmm.to_csv('XYZ.csv')


with open('Resnet_1.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["loss", "accuracy", "val_loss", "val_accuracy"])
    for i in range(len(hist.history['loss'])):
        writer.writerow([hist.history['loss'][i], hist.history['accuracy'][i], hist.history['val_loss'][i], hist.history['val_accuracy'][i]])

score = model.evaluate(E_test, r_test)
print('Test loss:', score[0] * 100)
print('Test accuracy:', score[1] * 100)

# Plot the training loss and accuracy


plt.figure(figsize=(6, 4))
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.ylim(0, 1)
plt.legend(['Train_Resnet', 'Validation_Resnet'], loc='upper left')
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.ylim(0, 1)
plt.legend(['Train_Resnet', 'Validation_Resnet'], loc='upper left')
plt.show()

# Load the Iris dataset with 7 features
# locatio = pd.read_csv(r"D:\Study folder\6 semester\ML\project\code\iris_2.csv")

# # Convert any float columns to string
# locatio = locatio.astype(str)

# N = locatio.iloc[:, :-1].values
# m = locatio.iloc[:, -1].values
# le = LabelEncoder()
# m = le.fit_transform(m)

# # Convert class labels to one-hot encoded vectors
# m = to_categorical(m)

# # Split the data into training and testing sets
# N_train, N_test, m_train, m_test = train_test_split(N, m, test_size=0.2)

# # Apply LabelEncoder separately to each column of m_train
# m_train_le = np.zeros_like(m_train).astype('float32')
# m_test_le = np.zeros_like(m_test).astype('float32')
# for i in range(m_train.shape[1]):
#     m_train_le[:, i] = LabelEncoder().fit_transform(m_train[:, i])
#     m_test_le[:, i] = LabelEncoder().fit_transform(m_test[:, i])

# # Apply LabelEncoder to N_train
# N_train_le = np.zeros_like(N_train).astype('float32')
# for i in range(N_train.shape[1]):
#     N_train_le[:, i] = LabelEncoder().fit_transform(N_train[:, i])

# # Apply LabelEncoder to N_test
# N_test_le = np.zeros_like(N_test).astype('float32')
# for i in range(N_test.shape[1]):
#     N_test_le[:, i] = LabelEncoder().fit_transform(N_test[:, i])

# # Create a simple neural network model
# model = Sequential([
#     Dense(16, activation='relu', input_shape=(7,)),
#     Dense(8, activation='relu'),
#     Dense(3, activation='softmax')
# ])
# # Compile the model
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model and store the history
# history = model.fit(N_train_le, m_train, epochs=100, batch_size=8, validation_data=(N_test_le, m_test))

# with open('Resnet_2.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(["loss", "accuracy", "val_loss", "val_accuracy"])
#     for i in range(len(history.history['loss'])):
#writer.writerow([history.history['loss'][i], history.history['accuracy'][i], history.history['val_loss'][i], history.history['val_accuracy'][i]])

# # Evaluate the model on the testing set
# score = model.evaluate(N_test_le, m_test)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

# # Plot the training and validation loss and accuracy
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train_Resnet', 'Validation_Resnet'], loc='upper right')
# plt.show()



