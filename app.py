from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)


try:
    model = joblib.load('customer_churn_model.pkl')
except Exception as e:
    print("Error loading the model:", e)

@app.route('/')
def home():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        age = request.form['Age']
        account_manager = request.form['Account_Manager']
        
       
        input_data = pd.DataFrame([[age, account_manager]], columns=['Age', 'Account_Manager'])
        
        
        prediction = model.predict(input_data)
        
       
        if prediction[0] == 1:
            result_message = 'This customer is likely to churn.'
        else:
            result_message = 'This customer is unlikely to churn.'
        
       
        return render_template('index.html', result=result_message)
    except Exception as e:
        print("Error during prediction:", e)
        return render_template('index.html', result="An error occurred during prediction.")

if __name__ == '__main__':
    app.run(debug=True)
