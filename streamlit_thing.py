# Created by Nafis Khan u3253332
import streamlit as st
import pandas as pd
import seaborn as sns
import requests
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
from PIL import Image
from risk_model import Prediction

# Create page
st.set_page_config(page_title="Heart Attack Risk Webpage", page_icon=":shooting_star:", layout="wide")

# Create navigation bar
selected = option_menu(
    menu_title=None,
    options=["Home", "Data", "About"],
    icons=["house", "clipboard-data", "person"],
    default_index=0,
    orientation="horizontal"
)


# Create gif function
def load_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Create home page
def home():
    with st.container():
        # Create 2 columns
        left_column, right_column = st.columns([2, 3], gap="medium")
        # Create title with description of heart attacks
        with left_column:
            st.title("Heart Attack Risk")
            st.write(
                "A heart attack, medically known as a myocardial infarction, is a serious and often life-threatening "
                "event that occurs when the blood flow to a part of the heart muscle is severely reduced or blocked, "
                "leading to damage or death of the heart tissue. This blockage is usually caused by the buildup of "
                "cholesterol, fat, and other substances, forming a plaque in the coronary arteries. When the blood flow "
                "is restricted, the affected part of the heart is deprived of oxygen, leading to chest pain or discomfort "
                "known as angina. If the blood flow isn't restored quickly, either by medical intervention or on its own, "
                "the heart muscle begins to die. The symptoms of a heart attack can vary widely, with some people "
                "experiencing intense chest pain, while others may have mild discomfort or feel nauseous, light-headed, "
                "or short of breath. Prompt medical attention is crucial during a heart attack, as timely intervention can"
                " minimize heart damage and improve the chances of survival.")

            st.write("")
            st.write("Preventing heart attacks involves managing risk factors such as high blood pressure, "
                     "high cholesterol, diabetes, obesity, and smoking. Adopting a heart-healthy lifestyle that includes"
                     " regular exercise, a balanced diet low in saturated fats and cholesterol, maintaining a healthy "
                     "weight, and avoiding tobacco products significantly reduces the risk. Additionally, understanding "
                     "the warning signs of a heart attack, which can also include pain or discomfort in the arms, back, "
                     "neck, jaw, or stomach, along with cold sweats, fatigue, and light-headedness, is crucial. Recognizing "
                     "these symptoms and seeking immediate medical help can make a vital difference in saving lives and "
                     "preserving heart health.")

        # Display GIF
        with right_column:
            asteroid_gif2 = load_url("https://lottie.host/dd0ed8c0-a8f0-4058-9773-116ebc58a5b8/9eQUq2Efil.json")
            st_lottie(asteroid_gif2, height=500, key="coding2")


# Create drop down menu to select EDA and PDA pages
def pages():
    page = st.sidebar.selectbox("Select Predict or Explore", ("Predict", "Explore"))
    if page == "Predict":
        predict_page()
    if page == "Explore":
        explore()


# Create prediction page(PDA)
def predict_page():
    # Create sidebar for user to put in information
    st.sidebar.header("User Input Parameters")
    Age = st.sidebar.slider("Age", 0, 0, 110)
    Cholesterol = st.sidebar.slider("Cholesterol Level", 0, 0, 500)
    Heart_Rate = st.sidebar.slider("Heart Rate", 0, 0, 400)
    Diabetes = st.sidebar.slider("Diabetes", 0, 0, 1)
    Family_History = st.sidebar.slider("Family History", 0, 0, 1)
    Smoking = st.sidebar.slider("Smoking", 0, 0, 1)
    Obesity = st.sidebar.slider("Obesity", 0, 0, 1)
    Exercise_Hours_Per_Week = st.sidebar.slider("Exercise Hours Per Week", 0, 0, 168)
    Previous_Heart_Problems = st.sidebar.slider("Previous Heart Problems", 0, 0, 1)
    Medication_Use = st.sidebar.slider("Medication_Use", 0, 0, 1)
    Stress_Level = st.sidebar.slider("Stress Level", 0, 0, 10)
    Sedentary_Hours_Per_Day = st.sidebar.slider("Sedentary Hours Per Day", 0, 0, 24)
    Income = st.sidebar.slider("Income", 0, 0, 500000)
    BMI = st.sidebar.slider("BMI", 0, 0, 200)
    Triglycerides = st.sidebar.slider("Triglycerides", 0, 0, 1500)
    Physical_Activity_Days_Per_Week = st.sidebar.slider("Physical Activity Days Per Week", 0, 0, 7)
    Sleep_Hours_Per_Day = st.sidebar.slider("Sleep Hours Per Day", 0, 0, 24)

    data = {"Age": Age,
            "Cholesterol": Cholesterol,
            "Heart Rate": Heart_Rate,
            "Diabetes": Diabetes,
            "Family History": Family_History,
            "Smoking": Smoking,
            "Obesity": Obesity,
            "Exercise Hours Per Week": Exercise_Hours_Per_Week,
            "Previous Heart Problems": Previous_Heart_Problems,
            "Medication Use": Medication_Use,
            "Stress Level": Stress_Level,
            "Sedentary Hours Per Day": Sedentary_Hours_Per_Day,
            "Income": Income,
            "BMI": BMI,
            "Triglycerides": Triglycerides,
            "Physical Activity Days Per Week": Physical_Activity_Days_Per_Week,
            "Sleep Hours Per Day": Sleep_Hours_Per_Day}

    features = pd.DataFrame(data, index=[0])

    # Create headings
    st.title("Heart Attack Risk Prediction 🫀")
    st.subheader(f"This prediction has an accuracy of: {Prediction.model_accuracy:.0%}")

    # Show user input from sliders
    st.subheader("User Input")
    st.write("Key => 0 : No  |  1 : Yes")
    df = features
    st.write(df)

    # Predict outcome
    risk_info = list(data.values())
    risk_prediction = Prediction.best_model.predict([risk_info])
    if risk_prediction == [0]:
        st.success("Prediction: You are healthy and are not at risk")

    else:
        st.error("Prediction: Warning! You are at Risk of getting a Heart Attack!")



# Create EDA page
def explore():
    # Create title
    st.title("Exploratory Data Analysis")
    df = pd.read_csv("heart_attack_prediction_dataset2.csv")

    # Display dataframe
    st.subheader("Heart Attack Data frame")
    st.dataframe(df)

    # Display bar chart
    st.write("---")
    st.subheader("Not at Risk vs At Risk")
    st.write("key: 0 = Not at Risk, 1 = At Risk")
    st.bar_chart(df.HeartAttackRisk.value_counts())

    # Display Age vs Heart Attack Risk
    st.write("---")
    st.subheader("Age Distribution by Heart Attack Risk")
    fig1, ax1 = plt.subplots(figsize=(15, 5), dpi=80)
    sns.kdeplot(data=df, x='Age', hue="HeartAttackRisk", shade=True, common_norm=False,
                palette=['#4a7aff', '#e68bbe'], ax=ax1)
    st.pyplot(fig1)
    st.write("The age of a person does not affect the risk of a heart attack.")

    # Display Heart Rate vs Heart Attack Risk
    st.write("---")
    st.subheader("Heart Rate Distribution by Heart Attack Risk")
    fig2, ax2 = plt.subplots(figsize=(15, 5), dpi=80)
    sns.kdeplot(data=df, x='Heart Rate', hue='HeartAttackRisk', common_norm=False,
                palette=['#4a7aff', '#e68bbe'], ax=ax2)
    st.pyplot(fig2)
    st.write("From the plot above it can be deduced that having a higher heart rate does not increase the risk heart attack.")

    # Display Cholesterol vs Heart Attack Risk
    st.write("---")
    st.subheader("Cholesterol Levels by Heart Attack Risk")
    fig3, ax3 = plt.subplots(figsize=(15, 5), dpi=80)
    sns.kdeplot(data=df, x='Cholesterol', hue="HeartAttackRisk", shade=True, common_norm=False,
                palette=['#4a7aff', '#e68bbe'], ax=ax3)
    st.pyplot(fig3)
    st.write("Considering how the density plot of the 'No at Risk' is so similar to 'At Risk', the Cholesterol levels of person does not contribute as a risk factor towards heart attack.")

    # Display Gender vs Heart Attack Risk
    st.write("---")
    st.subheader("Gender Distribution by Heart Attack Risk Male/Female")
    gender_risk = df.groupby(['Sex', 'HeartAttackRisk']).size().unstack().fillna(0)

    # Create a bar chart
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    gender_risk.plot(kind='bar', stacked=True, color=['#e68bbe', '#fde4f2'], ax=ax4)
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.title('Gender Distribution by Heart Attack Risk Male/Female')
    plt.xticks(rotation=0)
    plt.legend(['No Heart Attack', 'Heart Attack'])
    st.pyplot(fig4)
    st.write("Men are more likely to be at risk of heart attack than woman.")

    # Correlation Heatmap
    st.write("---")
    st.subheader("Correlation Between Variables")
    num_col = df.select_dtypes(include=['int64', 'float64'])
    corr_matrix = num_col.corr()
    fig5, ax5 = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax5)
    st.pyplot(fig5)

# About page
def about():
    # Create heading
    st.markdown("<h1 style='text-align: center; color: white;'>About Me</h1>", unsafe_allow_html=True)
    st.write("")
    with st.container():
        col1, col2, col3 = st.columns([1.75, 3, 1.8])
        with col1:
            st.write("")
        with col2:
            image = Image.open("IMG1.jpg")
            st.image(image, width=600)
            st.write("I am a student at UC studying a Bachelors in Information Technology "
                     "u3253332@uni.canberra.edu.au ")
        with col3:
            st.write("")


# Pages
if selected == "Home":
    home()
if selected == "About":
    about()
if selected == "Data":
    pages()