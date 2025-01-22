import streamlit as st
import pandas as pd
import numpy as np
import joblib
from fpdf import FPDF

# Load the trained model
svc = joblib.load('predict.pkl')

# Load necessary dataframes
precautions = pd.read_csv('symptom_precaution.csv')
# precautions.drop('Unnamed: 0', inplace=True, axis=1)
description = pd.read_csv('symptom_description.csv')
medication = pd.read_csv('symptom_medications.csv')
diets = pd.read_csv('diets.csv')
workout = pd.read_csv('workout.csv')
# workout.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'], inplace=True, axis=1)
workout.rename(columns={'disease': 'Disease'}, inplace=True)

symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of_urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

def get_assistance(disease):
  descr = description[description['Disease'] == disease]['Description'].values[0]
  prec = precautions[precautions['Disease'] == disease][['Precaution_1','Precaution_2','Precaution_3','Precaution_4']].values[0]
  medic = medication[medication['Disease'] == disease]['Medication'].values[0]
  diet = diets[diets['Disease'] == disease]['Diet'].values[0]
  work = workout[workout['Disease'] == disease]['workout']
  return descr,prec,medic,diet,work

def predict_disease(symptoms):
  input_data = np.zeros(len(symptoms_dict))
  for symptom in symptoms:
    if symptom in symptoms_dict.keys():
      input_data[symptoms_dict[symptom]] = 1
  input_data = input_data.reshape(1,-1)
  prediction = svc.predict(input_data)
  predicted_disease = prediction[0]
  return diseases_list[predicted_disease]

def generate_pdf(disease, description, precautions, medications, diet, workout):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Title
    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(200, 10, txt="Healthcare Assistant Report", ln=True, align="C")
    pdf.ln(10)

    # Predicted Disease
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(0, 10, txt="Predicted Disease:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=disease)
    pdf.ln(5)

    # Description
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(0, 10, txt="Disease Description:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=description)
    pdf.ln(5)

    # Precautions
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(0, 10, txt="Precautions:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt="\n".join(precautions))
    pdf.ln(5)

    # Medications
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(0, 10, txt="Medications:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=medications)
    pdf.ln(5)

    # Diet
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(0, 10, txt="Dietary Recommendations:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=diet)
    pdf.ln(5)

    # Workout
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(0, 10, txt="Workout Recommendations:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt="\n".join(workout))

    return pdf

def display_home():
  st.title("Welcome to HealthIQ :AI Powered Smart Healthcare Assistant")
  st.write(
  """This project is an AI-powered healthcare assistant 
    designed to provide a holistic approach to managing health-related 
    concerns. It leverages machine learning and data analytics to predict
    possible diseases based on user-entered symptoms and offers comprehensive
    insights for better healthcare management.""")
  st.markdown("""
  This app is designed to provide a personalized healthcare experience by analyzing symptoms and offering detailed insights. With just a few clicks, you can:

  - **Predict Diseases:** Enter your symptoms to receive a prediction of the most likely condition.
  - **Understand Your Health Condition:** Access a comprehensive description of the predicted disease to learn more about it.
  - **Get Precautions and Medications:** Receive tailored precautions and suggested medications to manage or prevent complications.
  - **Customized Diet Plans:** Explore dietary recommendations designed to aid recovery and maintain good health.
  - **Workout Suggestions:** View exercises and workouts suitable for your condition to promote overall well-being.

  This tool is developed using advanced machine learning models and extensive medical datasets to assist users in making informed health decisions.

  ### How It Works
  1. Enter your symptoms in the provided fields.
  2. Get the predicted disease along with actionable insights.
  3. Download a PDF report summarizing all the details for future reference.

  ---

  > **Note:** This app is intended for informational purposes only and should not replace professional medical advice. Always consult a healthcare provider for medical concerns.

  ---

  We value your feedback! Feel free to share suggestions to improve the app further.
  """)

def Health_Care():
  st.title("HealthIQ :AI Powered Smart Healthcare Assistant")
  st.write(
        """
        **Predict your Disease**: Select your symptoms, and the AI model will predict the possible disease.
        You will receive tailored advice on how to manage your health condition.
        """
  )

  # Symptom selection input
  symptoms = st.multiselect(
        "Please Select your symptoms:",
        symptoms_dict.keys(),
        help="Select one or more symptoms you are experiencing."
  )

  if st.button("Predict Disease"):
    if not symptoms:
      st.warning("⚠️ Please select at least one symptom to proceed.")
    else:
      predicted_disease = predict_disease(symptoms)
      st.success(f"🩺 The predicted disease is: **{predicted_disease}**")
      description, precautions, medications, diet, workout = get_assistance(predicted_disease)
      # Display additional info
      st.subheader("📄 Description")
      st.write(f"📝 {description}")

      st.subheader("🚨 Precautions")
      for prec in precautions:
        st.write(f"{prec}")

      st.subheader("💊 Medications")
      st.write(f"{medications}")
      st.markdown(
          "<p style='color:red;'><b>Before taking this medicine, please visit the clinic or hospital. Thank you.</b></p>",
          unsafe_allow_html=True
      )

      st.subheader("🥗 Recommended Diet")
      st.write(f"{diet}")

      st.subheader("🏋️ Recommended Workouts")
      for i in workout:
        st.write(f"{i}")
        
  ##Generate PDF
  if st.button("Generate PDF"):
    predicted_disease = predict_disease(symptoms)
    description, precautions, medications, diet, workout = get_assistance(predicted_disease)
    pdf = generate_pdf(predicted_disease, description, precautions, medications, diet, workout)
    pdf_file = "Healthcare_Report.pdf"
    pdf.output(pdf_file)

    with open(pdf_file, "rb") as f:
        st.download_button(
            label="Download PDF",
            data=f,
            file_name=pdf_file,
            mime="application/pdf",
        )
  # Footer
  st.markdown("---")
  st.write("🔧 Developed by **VEERA RAGHAVA REDDY**")

def display_about():
  st.title("About Us")
  st.write(
        """
        **Developer**: VEERA RAGHAVA REDDY

        **GitHub**: [GitHub](https://github.com/RAGHAVA-REDDY123)
        
        **LinkedIn**: [LinkedIn](www.linkedin.com/in/veeraraghavareddy123)
        """
  )
  st.markdown(
    """
    ### 🙏 Thank You for Using This App!  
    We hope this application provided you with valuable insights into your health.  
    Your feedback and suggestions are crucial for us to improve and add more features.  

    💡 **Got a suggestion?**  
    Drop your ideas or feedback at: [veeraraghavareddy2006@gmail.com](mailto:veeraraghavareddy2006@gmail.com)  
    Or connect with us on [LinkedIn](www.linkedin.com/in/veeraraghavareddy123)

    Let's make healthcare smarter and more accessible together!  
    """
  )
def main():
    st.set_page_config(page_title="HealthIQ :AI Powered Smart Healthcare Assistant", page_icon="🌟", layout="centered")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a page", ["Home", "Health-Care", "About Us"])

    # Conditional page routing
    if page == "Home":
        display_home()
    elif page == "Health-Care":
        Health_Care()
    elif page == "About Us":
        display_about()

if __name__ == "__main__":
    main()



