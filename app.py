import streamlit as st
import pickle
import pandas as pd
import os

# Ensure correct path to pipe.pkl
pipe_path = os.path.join(os.path.dirname(__file__), "pipe.pkl")

# Load model safely
if os.path.exists(pipe_path):
    with open(pipe_path, "rb") as f:
        pipe = pickle.load(f)
else:
    st.error("Model file 'pipe.pkl' not found. Please upload it.")

st.title('IPL Win Predictor')

teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
         'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
         'Rajasthan Royals', 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

col1, col2 = st.columns(2)  # ✅ FIXED

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

selected_city = st.selectbox('Select host city', sorted(cities))
target = st.number_input('Target', min_value=1)

col3, col4, col5 = st.columns(3)  # ✅ FIXED

with col3:
    score = st.number_input('Score', min_value=0)
with col4:
    overs = st.number_input('Overs completed', min_value=0.0, max_value=20.0, step=0.1)
with col5:
    wickets_out = st.number_input('Wickets out', min_value=0, max_value=10)

if st.button('Predict Probability'):
    if not os.path.exists(pipe_path):
        st.error("Model file not found! Upload 'pipe.pkl' to the project directory.")
    else:
        runs_left = target - score
        balls_left = 120 - int(overs * 6)
        wickets = 10 - wickets_out

        # Prevent ZeroDivisionError
        crr = score / overs if overs > 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        result = pipe.predict_proba(input_df)
        loss_prob = result[0][0]
        win_prob = result[0][1]

        st.header(f"{batting_team} - {round(win_prob * 100)}%")
        st.header(f"{bowling_team} - {round(loss_prob * 100)}%")
