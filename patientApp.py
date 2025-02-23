import pandas as pd
import streamlit as st
from openai import OpenAI
from PIL import Image

img = Image.open("logo.png")
st.set_page_config(
    page_title="PhysIQ - At Home AI Diagnosis",
    page_icon=img,
    layout="wide",
)

st.markdown(
    """
    <style>
    /* Overall page styling */
    body {
        background-color: #f0f2f6;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #333;
        font-size: 50px;
    }
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 30px 0 10px 0;
    }
    /* Container for main content */
    .main-container {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 30px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        margin: 20px auto;
    }
    /* Button styling */
    .stButton button {
        background-color: #623ea8;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        color: #fff;
        font-size: 16px;
        cursor: pointer;
        margin-top: 20px;
    }
    .stButton button:hover {
        background-color: #3c2569;
        color: #fff;
        text-color: #fff !important;
    }
    .stMultiSelect div[role="button"] span {
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Load the dataset to obtain a list of possible symptoms
train_data = pd.read_csv("C:/Users/User/Downloads/patient_data/Disease_symptom_and_patient_profile_dataset.csv")
train_data.columns = [x.lower() for x in train_data.columns]

# Assume that columns other than 'disease' and 'outcome variable' represent symptoms
symptoms = list(train_data.columns)

st.title("PhysIQ")
st.header("At home AI diagnosis")

# Let user select symptoms from the list
user_symptoms = st.multiselect("Select your symptoms:", symptoms)

if st.button("Diagnose"):
    if user_symptoms:
        # Build a prompt for GPT‑4 using the selected symptoms
        prompt = (
            f"I have the following symptoms: {', '.join(user_symptoms)}. "
            "Based on these, what disease or medical condition might I have? "
            "Please provide a brief and concise diagnosis. Keep the answers very brief and casual as to not scare patients. Provide advice and at-home possible remedies and advise to see doctors as needed."
        )
        
        # Set your OpenAI API key (it is recommended to store this securely in Streamlit secrets)
        client = OpenAI(
    api_key= "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"  # This is the default and can be omitted
)

        # Call the GPT‑4 API with the prompt
        chat_completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful and knowledgeable medical assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        
        # Extract and display the predicted diagnosis
        predicted_disease =  chat_completion.choices[0].message.content.strip()
        st.subheader("Predicted Disease:")
        st.write(predicted_disease)
    else:
        st.write("Please select at least one symptom.")