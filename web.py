import os
import joblib
from flask import Flask, request, redirect, url_for, flash
import pymysql
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from transformers import BertTokenizer  # Import BERT tokenizer
import numpy as np

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Load the pre-trained model
model = joblib.load('svm_bert_model.pkl')

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Use the same tokenizer as during training

# Database configuration
db_config = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'db': os.getenv('DB_NAME'),
    'cursorclass': pymysql.cursors.DictCursor
}

# Email configuration
EMAIL_USER = os.getenv('EMAIL_USER')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')
EMAIL_RECIPIENT = os.getenv('EMAIL_RECIPIENT')

def send_email(ip_address, malicious_input):
    subject = "Malicious Input Detected"
    body = f"Malicious input detected from IP: {ip_address}\nInput: {malicious_input}"
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = EMAIL_USER
    msg['To'] = EMAIL_RECIPIENT

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_USER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_USER, EMAIL_RECIPIENT, msg.as_string())
        print("Email sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")

def check_user_in_db(username, password):
    connection = None  # Initialize connection
    try:
        connection = pymysql.connect(**db_config)
        with connection.cursor() as cursor:
            # First, check if the username exists
            sql_user_check = "SELECT * FROM users WHERE user_id = %s"
            cursor.execute(sql_user_check, (username,))
            user = cursor.fetchone()
            
            if not user:
                return None  # Username does not exist
            
            # If username exists, check for password match
            sql_password_check = "SELECT * FROM users WHERE user_id = %s AND password = %s"
            cursor.execute(sql_password_check, (username, password))
            user_with_password = cursor.fetchone()

            if user_with_password:
                return True  # Username and password match
            else:
                return False  # Username exists, but password is incorrect

    except Exception as e:
        print(f"Database error: {e}")
        return None
    finally:
        if connection:
            connection.close()

@app.route('/')
def home():
    return redirect(url_for('login_page'))

@app.route('/login_page', methods=['GET'])
def login_page():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Login Page</title>
        <style>
            body {
                font-family: 'Arial', sans-serif;
                background-color: #f0f2f5; /* Light gray background */
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
            }

            /* Container Styles */
            .container {
                background: white;
                padding: 40px;
                border-radius: 8px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
                width: 400px; /* Fixed width for the form */
                text-align: center; /* Center text */
            }

            /* Header Styles */
            h2 {
                margin-bottom: 30px;
                color: #333; /* Darker text color */
            }

            /* Form Group Styles */
            .form-group {
                margin-bottom: 20px;
                text-align: left; /* Align labels to the left */
            }

            /* Label Styles */
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold; /* Bold labels */
                color: #555; /* Medium gray color */
            }

            /* Input Styles */
            input[type="text"],
            input[type="password"] {
                width: 100%;
                padding: 12px;
                border: 1px solid #ccc;
                border-radius: 4px;
                transition: border-color 0.3s; /* Smooth transition for border color */
            }

            /* Input Focus Styles */
            input[type="text"]:focus,
            input[type="password"]:focus {
                border-color: #007bff; /* Blue border on focus */
                outline: none; /* Remove default outline */
            }

            /* Button Styles */
            button {
                padding: 12px 20px;
                background-color: #007bff; /* Bootstrap primary color */
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
                transition: background-color 0.3s; /* Smooth transition for background color */
            }

            /* Button Hover Styles */
            button:hover {
                background-color: #0056b3; /* Darker blue on hover */
            }

            /* Footer Styles */
            .footer {
                margin-top: 20px;
                font-size: 14px;
                color: #777; /* Light gray color */
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Login</h2>
            <form action="/login" method="POST">  <!-- Updated action to /login -->
                <div class="form-group">
                    <label for="username">Username:</label>  <!-- Updated field name to username -->
                    <input type="text" id="username" name="username" required>
                </div>
                <div class="form-group">
                    <label for="password">Password:</label>  <!-- Updated field name to password -->
                    <input type="password" id="password" name="password" >
                </div>
                <button type="submit">Login</button>
            </form>
        </div>
    </body>
    </html>
    """

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    user_input = f"Username: {username}, Password: {password}"

    # Preprocess the input using the BERT tokenizer
    inputs = tokenizer(
        user_input,                      
        return_tensors='pt',             
        padding=True,                    
        truncation=True,                 
        max_length=128                   
    )
    input_ids = inputs['input_ids'].numpy()
    attention_mask = inputs['attention_mask'].numpy()

    # Flatten the input to 2D for the SVC model
    input_ids_flattened = input_ids.reshape(1, -1)
    attention_mask_flattened = attention_mask.reshape(1, -1)

    # Combine input_ids and attention_mask into a single feature vector
    combined_features = np.concatenate([input_ids_flattened, attention_mask_flattened], axis=1)

    # Pad or truncate the feature vector
    expected_features = 772  # Adjust according to the trained model
    if combined_features.shape[1] < expected_features:
        padding = np.zeros((1, expected_features - combined_features.shape[1]))
        combined_features = np.concatenate([combined_features, padding], axis=1)
    elif combined_features.shape[1] > expected_features:
        combined_features = combined_features[:, :expected_features]

    # Predict if the input is malicious
    prediction = model.predict(combined_features)[0]

    ip_address = request.remote_addr

    if prediction == 1:  # Malicious input detected
        send_email(ip_address, f"Malicious attempt detected! Input: {user_input}")
        flash("Malicious activity detected. Administrators have been notified.")
        return redirect(url_for('login_page'))

    # Check if user exists in the database
    login_status = check_user_in_db(username, password)

    if login_status is True:
        print("User authenticated successfully. Redirecting to dummy page.")
        return redirect(url_for('dummy_page'))
    elif login_status is False:
        send_email(ip_address, f"Failed login attempt: Username exists, but incorrect password entered.\nUsername: {username}")
        flash("Incorrect password. If this wasn't you, please check your account security.")
        return redirect(url_for('login_page'))
    else:
        flash("Invalid username or password.")
        return redirect(url_for('login_page'))

@app.route('/dummy_page')
def dummy_page():
    return "Welcome to the dummy page!"

if __name__ == '__main__':
    app.run(debug=True)