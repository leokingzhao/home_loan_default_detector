import pickle

from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

# Create flask app
app = Flask(__name__)

# Load the pickle model
model = pickle.load(open('model.pkl', 'rb'))


# Load the training data into a pandas DataFrame (replace 'data.csv' with your actual dataset)
df = pd.read_csv('application_train.csv')

# List of categorical features that need label encoding
cat_features = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'NAME_EDUCATION_TYPE', 'FLAG_OWN_CAR']

# Create a dictionary to store the label encoders
label_encoders = {}

# Create and fit label encoders for each categorical feature
for cat_feat in cat_features:
    le = LabelEncoder()
    le.fit(df[cat_feat])
    label_encoders[cat_feat] = le

# Save the label encoders into 'label_encoders.pkl'
with open('label_encoders.pkl', 'wb') as file:
    pickle.dump(label_encoders, file)





@app.route("/")
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    # Get form data
    data = request.form.to_dict()

    # Prepare numerical features
    num_features = ['REGION_POPULATION_RELATIVE', 'EXT_SOURCE_3', 'DAYS_LAST_PHONE_CHANGE']
    numerical_values = [float(data[feat]) for feat in num_features]
    numerical_features = np.array(numerical_values)

    # Prepare categorical features
    cat_features = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'NAME_EDUCATION_TYPE', 'FLAG_OWN_CAR']
    categorical_values = [data[feat] for feat in cat_features]
    categorical_features = np.array(categorical_values)

    # Convert categorical features using label encoders
    for i, cat_feat in enumerate(cat_features):
        categorical_features[i] = label_encoders[cat_feat].transform([categorical_features[i]])[0]

    # Combine numerical and categorical features
    features = np.concatenate((numerical_features, categorical_features))

    # Prepare the features for prediction
    prediction = model.predict([features])

    return render_template('index.html', prediction_text='The customer is {}'.format(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True)
