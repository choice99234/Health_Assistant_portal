from flask import Flask, render_template, request, redirect, session, flash, url_for,jsonify
from flask_login import current_user
from flask_sqlalchemy import SQLAlchemy
import bcrypt
import secrets
import json
import os
import requests
from fuzzywuzzy import process

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your_default_secret_key')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'your_email@gmail.com'
app.config['MAIL_PASSWORD'] = 'your_email_password'
port = int(os.environ.get("PORT", 5000))  # Default to 5000 if PORT isn't set

db = SQLAlchemy(app)
# Predefined admin credentials

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=True)
    password = db.Column(db.String(100))

    def __init__(self, username, email, password):
        self.username = username
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))

with app.app_context():
    db.create_all()

@app.route('/')
def index():
    return render_template('base.html')

# Importing the check_authenticated decorator
from functools import wraps

# Definition of the check_authenticated decorator
def check_authenticated(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if 'email' in session:
            return redirect('/dashboard')
        else:
            return func(*args, **kwargs)
    return wrapper

# Applying the check_authenticated decorator to the register route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        # Check if the email is already in use
        existing_user = User.query.filter_by(email=email).first()
        
        if existing_user:
            flash('User with the same email already exist. Please choose a different email.')
            return render_template('register.html')

        if not username or not email or not password:
            flash('All fields are required', 'error')
            return redirect('/register')
    
        # If email is not in use, proceed with registration
        new_user = User(username=username, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful. Please login.', 'success')
        return render_template('login.html')

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Check if the user exists in the database
        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            session['username'] = user.username  # Store the username in the session
            session['email'] = user.email
            session['last_activity'] = datetime.now()  # Update last activity timestamp
            flash('Login was successful!', 'success')  # Add success category for flash
            return redirect('dashboard')

        flash('Invalid username or password!', 'error')  # Add error category for flash
        return redirect(url_for('login'))

    return render_template('login.html')


from datetime import datetime, timedelta

# Definition of the check_authenticated decorator
def check_authenticated(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if 'email' in session:
            # Check if the session has expired
            if 'last_activity' in session:
                last_activity_time = session['last_activity']
                session_timeout = timedelta(minutes=2)  # Adjust as needed
                if datetime.now() - last_activity_time > session_timeout:
                    flash('Your session has expired. Please log in again.', 'error')
                    session.clear()  # Clear session data
                    return redirect('/login')

                # Update last activity timestamp
                session['last_activity'] = datetime.now()
            return func(*args, **kwargs)
        else:
            flash('You need to login to access this page.', 'error')
            return redirect('/login')
    return wrapper

# Define a function to check if the user is authenticated
def check_authenticated(func):
    def wrapper(*args, **kwargs):
        if 'email' in session:
            return func(*args, **kwargs)
        else:
            flash('You need to login to access this page.', 'error')
            return redirect('/login')
    return wrapper

# Apply the authentication check decorator to routes that should be restricted to logged-in users
@app.route('/dashboard')
@check_authenticated
def dashboard():
    if 'email' in session:  # Check if 'email' is in session
        user = User.query.filter_by(email=session['email']).first()  # Retrieve the user
        return render_template('dashboard.html', user=user)  # Pass user to template
    return redirect('/base')

@app.route('/logout')
def logout():
    session.pop('email', None)
    flash('You are now logged out!')
    return render_template('base.html')

# 4. Handle Password Reset Route
@app.route('/reset/<token>', methods=['GET', 'POST'])
def reset(token):
    if request.method == 'POST':
        new_password = request.form['new_password']
        confirm_password = request.form['confirm_password']

        # Add your logic to verify the token and update the password
        if new_password == confirm_password:
            # Update the password in the database for the user associated with the token
            flash('Password reset successfully. You can now login with your new password.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Passwords do not match. Please try again.', 'error')
            return redirect(url_for('reset', token=token))

    # Render the password reset form
    return render_template('reset.html', token=token)



#=======================================chatbot====================================================================
import nltk
from chatbot import bot_response, greeting_response, get_corpus, get_sentence_list

# Route to render chatbot interface
@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

# Get sentence list for bot processing
sentence_list = get_sentence_list()

# Route to process user input and return bot response
@app.route('/get', methods=['POST'])
def get_bot_response():
    user_input = request.form['msg']
    
    # Check if it's a greeting
    greeting = greeting_response(user_input)
    if greeting:
        return greeting

    # Otherwise, generate a response from the bot
    response = bot_response(user_input, sentence_list)
    return response


#===============================================================Medicine Recommendation System backend code======================================================================
import numpy as np
import  pandas as pd
import pickle
import difflib


#Load Databases=====================================================================================
precautions=pd.read_csv('datasets/precautions_df.csv')
workout =pd.read_csv('datasets/workout_df.csv')
description=pd.read_csv('datasets/description.csv')
medications  =pd.read_csv('datasets/medications.csv')
diets =pd.read_csv('datasets/diets.csv')


#Load Model===================================================================================
Model=pickle.load(open("predictor/Model.pkl", 'rb'))

# Helper function
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = pre.values.flatten().tolist()  # Convert to flat list

    med = medications[medications['Disease'] == dis]['Medication']
    med = med.str.split(',').tolist()  # Split the string and convert to list

    die = diets[diets['Disease'] == dis]['Diet']
    die = die.str.split(',').tolist()  # Split the string and convert to list

    wrkout = workout[workout['disease'] == dis]['workout']
    wrkout = wrkout.tolist()  # Convert to list

    return desc, pre, med, die, wrkout



symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}


#Model prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))

    for item in patient_symptoms:
        input_vector[symptoms_dict[item]]= 1

    return diseases_list[Model.predict([input_vector])[0]]


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms').strip()

        if not symptoms:
            flash('Please enter symptoms', 'error')
            return redirect('/dashboard')

        user_symptoms = [s.strip() for s in symptoms.split(',')]
        user_symptoms = [sym.strip("[]' ") for sym in user_symptoms]

        corrected_symptoms = []
        invalid_symptoms = []

        for sym in user_symptoms:
            close_matches = difflib.get_close_matches(sym, symptoms_dict.keys(), n=1, cutoff=0.8)
            if close_matches:
                corrected_symptoms.append(close_matches[0])
            else:
                invalid_symptoms.append(sym)

        if len(invalid_symptoms) > 5:
            flash(f"Many invalid symptoms entered: {', '.join(invalid_symptoms)}. It is better to consult a doctor when your health is at risk of being weakened.", 'error')
            return redirect('/dashboard')
        elif invalid_symptoms:
            flash("It is better to consult a doctor when your health is at risk of being weakened.", 'warning')

        if not corrected_symptoms:
            flash('No valid symptoms entered for prediction.', 'error')
            return redirect('/dashboard')

        try:
            predicted_disease = get_predicted_value(corrected_symptoms)
        except Exception as e:
            flash('An error occurred during prediction. Please try again.', 'error')
            return redirect('/dashboard')

        desc, pre, med, die, wrkout = helper(predicted_disease)

        # Flatten lists of lists
        my_med = [item for sublist in med for item in sublist]
        my_die = [item for sublist in die for item in sublist]

        return render_template('dashboard.html', predicted_disease=predicted_disease, dis_des=desc, dis_pre=pre, dis_med=my_med, dis_wrkout=wrkout, dis_die=my_die, user=current_user)

@app.route('/base')
def base():
    return render_template('base.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/help')
def help():
    return render_template('help.html')


@app.route('/developer')
def developer():
    return render_template('developer.html')



@app.route('/about')
def about():
    return render_template('about.html')



@app.route('/delete/<int:user_id>', methods=['POST'])  # Ensure POST method is allowed
def delete_user(user_id):
    user = User.query.get(user_id)
    if user:
        db.session.delete(user)
        db.session.commit()
        flash('User deleted successfully!', 'success')
    else:
        flash('User not found!', 'error')
    return redirect(url_for('admin_dashboard'))  # Redirect back to the admin dashboard


@app.route('/edit/<int:user_id>', methods=['GET', 'POST'])
def edit_user(user_id):
    user = User.query.get_or_404(user_id)  # Fetch the user or return 404
    if request.method == 'POST':
        user.username = request.form['username']
        user.email = request.form['email']
        user.password = request.form['password']  # Remember to hash this!
        db.session.commit()
        flash('User updated successfully!', 'success')
        return redirect(url_for('admin_dashboard'))
    return render_template('edit_user.html', user=user)  # Pass user to the template

@app.route('/search', methods=['POST'])
def search_user():
    username = request.form['username']
    users = User.query.filter_by(username=username).all()  # Adjust the query as necessary
    
    if users:
        return render_template('admin_dashboard.html', users=users)
    else:
        flash('No users found with that username.', 'warning')
        return redirect(url_for('admin_dashboard'))  # Redirect back to the admin dashboard


















def load_disease_data():
    try:
        # Check if the file exists
        if not os.path.exists('disease_data.json'):
            print("Error: disease_data.json file not found!")
            return {}

        # Try opening and loading the JSON file
        with open('disease_data.json', 'r') as file:
            disease_data = json.load(file)
            print("Disease data loaded successfully.")
            return disease_data

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return {}
    except Exception as e:
        print(f"An error occurred: {e}")
        return {}

# Load disease data
disease_info = load_disease_data()

@app.route('/disease')
def disease():
    return render_template('disease.html')

@app.route('/search_disease', methods=['POST'])
def search_disease():
    data_type = request.form.get('data_type')
    query = request.form.get('query')
    country = request.form.get('country', '').strip()
    disease_name = request.form.get('disease', '').strip().lower()  # Convert disease name to lowercase

    # Ensure disease data is only accessible for Malawi
    if country and country.lower() != 'malawi' and data_type == 'disease':
        return jsonify({"error": "Error: Data accessed here is from Malawi only."}), 400

    # Check if user wants information on a specific disease
    if data_type == 'disease' and disease_name:
        # Attempt to find the closest match using fuzzywuzzy
        matched_disease, score = process.extractOne(disease_name, disease_info.keys())

        # If the match score is above a threshold (e.g., 70%), return the matched disease info
        if score > 70:
            return jsonify(disease_info[matched_disease])
        else:
            return jsonify({"error": f"No matching disease found for '{disease_name}'."}), 404

    # If no disease query is given, or user is searching for COVID-19 data, use the API
    if data_type == 'global':
        url = "https://disease.sh/v3/covid-19/all"
    elif data_type == 'country':
        url = f"https://disease.sh/v3/covid-19/countries/{query}"
    elif data_type == 'continent':
        url = f"https://disease.sh/v3/covid-19/continents/{query}"
    elif data_type == 'state':
        url = f"https://disease.sh/v3/covid-19/states/{query}"
    else:
        return jsonify({"error": "Invalid data type or insufficient data provided."}), 400

    # Fetch COVID-19 data
    response = requests.get(url)
    if response.status_code == 200:
        return jsonify(response.json())
    else:
        return jsonify({"error": f"No data available for '{query}' in '{data_type}'."}), 404    

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=port)  # Binding to 0.0.0.0 allows external connections