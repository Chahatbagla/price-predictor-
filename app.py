from flask import Flask, request, render_template, redirect, url_for, jsonify
import joblib
import pandas as pd
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and data
model = joblib.load('random_forest_model.pkl')         # Ensure the model file path is correct
df_cleaned = pd.read_csv('your_data.csv')         # Load your dataset, e.g., `df_cleaned`
# Load your fitted Yeo-Johnson transformer
yeo_johnson_transformer = joblib.load('yeo_johnson_transformer.pkl')

# Define the home route with product name input form
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the product name from the form
        product_name = request.form['product_name']
        
        # Search for the product in the dataset
        product_info = df_cleaned[df_cleaned['name'].str.contains(product_name, case=False, na=False)]
        
        if not product_info.empty:
            # Select the first match
            product_details = product_info.iloc[0].to_dict()
            return render_template('product_info.html', product_details=product_details)
        else:
            return "Product not found. Please try another product name.", 404
    
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get product details from the request (using hidden fields from product_info page)
    product_details = request.form.to_dict()
    
    # Create the input DataFrame with the specified feature names and order
    input_data = pd.DataFrame([{
        'ratings': float(product_details['ratings']),        
        'no_of_ratings': float(product_details['no_of_ratings']),  
        'actual_price': float(product_details['actual_price']),    
        'rating_ratio': float(product_details['rating_ratio']),      
        'log_actual_price': float(product_details['log_actual_price']),    
    }])

    # Ensure the order of input_data matches the training features
    input_data = input_data[['ratings', 'no_of_ratings', 
                             'actual_price', 'rating_ratio', 
                             'log_actual_price']]
    
    # Perform prediction
    prediction = model.predict(input_data)

    # Apply inverse transformations
    predicted_discount_price = yeo_johnson_transformer.inverse_transform(prediction.reshape(-1, 1))

    # Display the result on a new page 
    return render_template('prediction_result.html', predicted_price=predicted_discount_price[0][0])


if __name__ == '__main__':
    app.run(debug=True)