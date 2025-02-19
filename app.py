import streamlit as st
from PIL import Image
import pandas as pd
import os
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import speech_recognition as sr
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv(r'Dataset/Medical_transcreption.csv')

# Text processing function
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return tokens

# Apply text cleaning
df['cleaned_symptoms'] = df['Symptoms'].apply(clean_text)

# Streamlit app
def main():
    st.title("Voice Based Medical Transcription")

    # Function to convert speech to text
    def speech_to_text():
        recognizer = sr.Recognizer()  # Reinitialize the recognizer to clear cache
        with sr.Microphone() as source:
            st.write("Adjusting for ambient noise... Please wait.")
            recognizer.adjust_for_ambient_noise(source)
            st.write("Listening... Speak now.")
            try:
                audio = recognizer.listen(source)
                text = recognizer.recognize_google(audio)
                st.write("You said: ", text)

                #result = df[df['Disease'].str.lower() == text.lower()]
                #st.write("result: ", result)
                #if not result.empty:
                
                result = df
                #st.write("result:", result)

                if 'dengue' in text.lower():
                    image = Image.open('images/dengue.png')
                    st.image(image, caption='dengue', use_column_width=True)
                    a=0
                elif 'malaria' in text.lower(): 
                    image = Image.open('images/malaria.jpg')
                    st.image(image, caption='Malaria', use_column_width=True)
                    a=1

                elif 'chikungunya' in text.lower():
                    image = Image.open('images/chikungunya.jpg')
                    st.image(image, caption='Chikungunya', use_column_width=True)
                    a=2
                elif 'corona' in text.lower():
                    image = Image.open('images/corona.jpg')
                    st.image(image, caption='Corona', use_column_width=True)
                    a=3
                    
                elif 'depression' in text.lower():
                    image = Image.open('images/depression.jpeg')
                    st.image(image, caption='Depression', use_column_width=True)
                    a=4

                elif 'anxiety' in text.lower():
                    image = Image.open("images/anxiety.jpg")
                    st.image(image, caption='Anxiety', use_column_width=True)
                    a=5

                elif 'diabetes' in text.lower():
                    image = Image.open('images/diabetes.webp')
                    st.image(image, caption='Diabetes', use_column_width=True)
                    a=6

                elif 'hypertension' in text.lower():
                    image = Image.open('images/hypertension.jpg')
                    st.image(image, caption='Hypertension', use_column_width=True)
                    a=7

                elif 'asthma' in text.lower():
                    image = Image.open("images/asthma.jpeg")
                    st.image(image, caption='Asthma', use_column_width=True)
                    a=8

                else:
                    st.write("Sorry, I didn't understand that. Please try again.") 
                
                #else:
                    #st.write("Disease not found")
                symptoms = result['Symptoms'].values[a]
                treatment = result['Treatment'].values[a]
                testings = result['Testings'].values[a]
                st.write(f"Symptoms: {symptoms}")
                st.write(f"Treatment: {treatment}")
                st.write(f"Testings: {testings}")    
            except sr.UnknownValueError:
                st.write("Speech Recognition could not understand the audio.")
            except sr.RequestError as e:
                st.write(f"Could not request results from Google Speech Recognition service; {e}")

    if st.button("Start Transcription"):
        speech_to_text()

if __name__ == "__main__":
    main()
