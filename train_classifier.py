# Import necessary libraries
import pickle  # For serializing and de-serializing Python objects
from sklearn.ensemble import RandomForestClassifier  # For creating a Random Forest Classifier model
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearn.metrics import accuracy_score  # For evaluating the model performance
import numpy as np  # For numerical operations

# Load the data from 'data.pickle'
data_dict = pickle.load(open('Models\data.pickle', 'rb'))

# Convert data and labels to NumPy arrays
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, shuffle=True, stratify=labels)

# Initialize a Random Forest Classifier model
model = RandomForestClassifier()

# Train the model using the training data
model.fit(x_train, y_train)

# Make predictions on the testing data
y_predict = model.predict(x_test)

# Calculate the accuracy of the model
score = accuracy_score(y_predict, y_test)

# Print the accuracy score
print('{}% of samples were classified correctly !'.format(score * 100))

# Save the trained model to a pickle file
f = open('Models\model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
