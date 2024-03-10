# Import necessary libraries
import streamlit as st  # Importing the Streamlit library for creating web applications
from streamlit_option_menu import option_menu
import numpy as np
import os
import pickle
from langchain.llms import OpenAI  # Importing the OpenAI language model
from langchain import PromptTemplate  # Importing a template for creating prompts
from langchain.chains import LLMChain  # Importing a chain to link language models
from langchain.memory import ConversationBufferMemory  # Importing memory for storing conversation history

# Import necessary libraries (duplicate import statement removed for clarity)
# Comment: The code imports essential libraries for data manipulation, machine learning, and interfacing with OpenAI language models.

# START UTILS FUNCTION

# load model, set cache to prevent reloading
# cache_data is used to cache anything which CAN be stored in a database (python primitives, dataframes, API calls)
# cache_resource is used to catche anything which CANNOT be stored in a database (ML models, DB connections)
# https://docs.streamlit.io/library/advanced-features/caching
@st.cache_resource
def load_diabetes_model():
    # load model using joblib
    with open('saved_model/diabetes_pipeline.pkl', 'rb') as dbpipe:
        diabetes_pipeline = pickle.load(dbpipe)

    return diabetes_pipeline

def calculate_and_interpret_bmi(weight_kg_value, height_feet_value):
    """
    Calculate BMI using the formula: BMI = weight (kg) / (height (m))^2
    """
    height_in_meters = height_feet_value / 3.281  # Convert height from feet to meters
    bmi = weight_kg_value / (height_in_meters ** 2)

    if bmi < 18.5:
        bmi_interpret =  "Under Weight"
        bmi_level = 0
    elif 18.5 <= bmi < 25:
        bmi_interpret = "Normal Weight"
        bmi_level = 0
    elif 25 <= bmi < 30:
        bmi_interpret = "Over Weight"
        bmi_level = 1 
    else:
        bmi_interpret = "Obese"
        bmi_level = 1

    return bmi, bmi_interpret, bmi_level

def calculate_desire_TDEE(activity_value,weight_kg_value,gender_value,age_value,height_feet_value):
    BMR=0.0
    TDEE=0.0

    # Convert height from feet to centimeters
    height_cms_value = 30.48 * height_feet_value

    # Used Mifflin - St Jeor's equation to calculate BMR
    # https://www.healthline.com/health/how-to-calculate-your-basal-metabolic-rate#limitations

    # Males: 10 × weight (in kilograms) + 6.25 × height (in centimeters) – 5 × age (in years) + 5
    # Females: 10 × weight (in kilograms) + 6.25 × height (in centimeters) – 5 × age (in years) – 161

    if gender_value == 0:
        BMR = (10 * weight_kg_value) + (6.25 * height_cms_value) - (5 * age_value) - 161
        #BMR=655+(weight_kg_value *9.6) + (1.8*height)-(4.7*age)
    else:
        #BMR=66+(bodyweight*13.7)+(5*height)-(6.8*age)
        BMR = (10 * weight_kg_value) + (6.25 * height_cms_value) - (5 * age_value) + 5
    
    # Get TDEE with respect to different daily activity 
    # 1.2: sedentary (little to no exercise)
    # 1.375: lightly active (light exercise 1 to 3 days per week)
    #1.55: moderately active (moderate exercise 6 to 7 days per week)
    #1.725: very active (hard exercise every day, or exercising twice a day)
    #1.9: extra active (very hard exercise, training, or a physical job)
    if activity_value == 0:
        TDEE = BMR * 1.2
    elif activity_value == 1:
        TDEE = BMR * 1.375
    elif activity_value == 2:
        TDEE = BMR * 1.55
    elif activity_value == 3:
        TDEE = BMR * 1.725    
    else:
        TDEE = BMR * 1.9

    return BMR, TDEE

def desired_nutrition(nutrition_goal_value,TDEE):
    # Get daily energy required due to different purpose
    if nutrition_goal_value==0 :   #cutting
        energy = TDEE - 300
    elif nutrition_goal_value==1 : #standard
        energy = TDEE
    else:  #Bulking
        energy=TDEE + 400

    # Carbohydrate(4 cal/gram) should account for 50% ,protein (4cal/gram) 25% and fat(9cal/gram) 25% 
    carbs = (energy * 0.5) / 4
    proteins =(energy * 0.25) / 4
    fats = (energy * 0.25) / 9  

    return energy,carbs,proteins,fats
# END UTIL FUNCTION

# hide deprication warnings which directly don't affect the working of the application
import warnings
warnings.filterwarnings("ignore")

# set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state (whether the page is loaded automatically or you need to perform some action for loading)
st.set_page_config(
    page_title="Diabetes Management System",
    initial_sidebar_state = 'auto',
    layout='wide'
)

# hide the part of the code, as this is just for adding some custom CSS styling but not a part of the main idea 
hide_streamlit_style = """
	<style>
    #MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
    div.block-container{padding-top:2rem;}
    div.stButton {text-align:center;}
    </style>
"""

# hide the CSS code from the screen as they are embedded in markdown text. 
# Also, allow streamlit to unsafely process as HTML
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

with st.sidebar:
        selected = option_menu('Diabetes Management System',
                              ['Diabetes Prediction and Nutrition','Diabetes Help Chatbot'],
                              icons=['prescription2','robot'],
                              menu_icon='hospital',
                              default_index =0  
                             )

if(selected == 'Diabetes Prediction and Nutrition'):
    st.title('Diabetes Prediction and Nutrition')

    # Input Boxes for user to input feature values
    yes_no_options = ['No', 'Yes']

    # Gender
    gender_options = ['Male','Female']
    gender = st.selectbox('Gender', gender_options)
    gender_value = 1 if gender == 'Male' else 0 

    # Age
    age_value = st.slider("Age (years):", 1, 80, 21)

    # Get weight and height for BMI
    weight_kg_value = st.slider("Enter your weight (kg)", 5, 100, value = 65)
    height_feet_value = st.slider("Enter your height (feet)", 1.0, 8.0, value = 5.9)
    
    bmi_value, bmi_interpret, bmi_level = calculate_and_interpret_bmi(weight_kg_value, height_feet_value)
    bmi_value_rounded = round(bmi_value, 2)
    
    st.write ("Your BMI is: **" + str(bmi_value_rounded) + "**")
    st.write("\n")

    # Hypertension
    hypertension = st.selectbox('Hypertension - Select Yest if currently diagnosed with high blood pressure else select No', yes_no_options)
    hypertension_value = 1 if hypertension == 'Yes' else 0 

    # Heart Disease
    heart_disease = st.selectbox('Heart Disease - Select Yes if currently diagnosed with any heart condition else select No', yes_no_options)
    heart_disease_value = 1 if heart_disease == 'Yes' else 0 

    #BMI
    #bmi_value = st.number_input("BMI - Body mass index (weight in kg/(height in m)^2)", min_value=10, max_value=100, value=20.00)

    # HbA1c_level
    hb_level_value = st.slider("HbA1c_level - Average blood sugar level over the past 2-3 months", 1.5, 9.5, 3.00)

    #blood_glucose_level
    blood_glucose_level_value = st.slider("Blood Glucose Level - Amount of glucose in the bloodstream at a given time", min_value=80, max_value=300, value=100)
    
    # Smoker / non-smoker
    smoker = st.selectbox('Smoker - If you are currently as smoker, select Yes else select No', yes_no_options)
    smoker_value = 1 if smoker == 'Yes' else 0 

    # Past smoker
    past_smoker = st.selectbox('Past Smoker - If you have smoked in the past, select Yes else select No', yes_no_options)
    past_smoker_value = 1 if past_smoker == 'Yes' else 0 

    # Daily activity selector
    activity_options = ['Sedentary (little to no exercise)',
                        'Lightly Active (light exercise 1 to 3 days per week)',
                        'Moderately Active (moderate exercise 6 to 7 days per week)',
                        'Very Active (hard exercise every day, or exercising twice a day)',
                        'Extra Active (very hard exercise, training, or a physical job)'
                        ]

    activity = st.selectbox('Select your level of daily activity', activity_options)
    activity_value = activity_options.index(activity) 
    
    nutrition_goal_options = ['Cutting',
                                'Standard',
                                'Bulking']
    nutrition_goal = st.selectbox('Select your desired nutrition goal', nutrition_goal_options)
    nutrition_goal_value  = nutrition_goal_options.index(nutrition_goal)

    st.write("\n")

    if st.button('View Prediction and Nutrition Recommendation'):
        # Gather all the numeric values read
        new_data_point = np.array([[age_value, hypertension_value, heart_disease_value, bmi_value, 
                                    hb_level_value, blood_glucose_level_value, gender_value, smoker_value, past_smoker_value]])

        # This will be called once as we have the decorator before the load_model function
        with st.spinner('Loading Model....'):
            # Load both the models
            diabetes_model = load_diabetes_model()

            # Make predictions using the loaded model
            prediction = diabetes_model.predict(new_data_point)

            corr_message = """
            Clinical correlation is strongly recommended to determine the significance of the finding.
            This will allow your doctor to make an accurate diagnosis using all available information  - your medical history,
            physical examination or any other laboratory tests.
            """
            
            # Display the prediction and corr message

            st.caption("Model Prediction")

            if prediction == 1:
                st.warning("**Diabetes**")
            else:
                st.success ("**No Diabetes**")

            st.write(corr_message)

            # Calculate your BMR and desired TDEE    
            BMR, desired_TDEE = calculate_desire_TDEE(activity_value,weight_kg_value,gender_value,age_value,height_feet_value)
            energy,carbs,proteins,fats = desired_nutrition(nutrition_goal_value,desired_TDEE)

            desired_TDEE = round(desired_TDEE)
            energy = round(energy)
            carbs = round(carbs)
            proteins = round(proteins)
            fats = round(fats)

            reco_message = "Your total daily energy expenditure (TDEE) calculated based on your activity volume is **" + str(desired_TDEE) +  "**." \
                           " To acheive your **" + str(nutrition_goal) + "** nutrition goal, you should have **" + str(energy) + " kcal** per day." \
                           " Your protein intake should be **" + str(proteins) + "** grams, carbs intake should be **" + str(carbs) + "** grams" \
                           " and your fat intake should be **" + str(fats) + "** grams."

            st.write("\n")
            
            st.caption("BMI Interpretation")
            if bmi_level == 0:
                st.success (bmi_interpret)
            else:
                st.warning (bmi_interpret)
            
            st.write("\n")
            
            st.caption("Nutrition Recommendation")
            st.success(reco_message)                

if(selected == 'Diabetes Help Chatbot'):
    # Set OpenAI API key from the constants file
    os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']

    # Set the title for the Streamlit web application
    st.title('Diabetes Help Bot')

    # Get user input for diabetic-related topics
    input_text = st.text_input("Ask about diabetic-related topics: Ask about diet? Blood Sugar Level? Complications?")

    # Define a template for the initial input prompt
    first_input_prompt = PromptTemplate(
        input_variables=['prompt'],
        template="reply this question {prompt} in the context of a diabetic patient."
    )

    # Set up memory for storing the conversation history
    person_memory = ConversationBufferMemory(input_key='prompt', memory_key='chat_history')

    # Create an instance of the OpenAI language model (LLM)
    llm = OpenAI(temperature=0.8)

    # Create a language model chain with the specified prompt template and memory
    chain = LLMChain(llm=llm, prompt=first_input_prompt, verbose=True)

    # Check if there is user input
    if input_text:
        # Run the language model chain with the user input and display the result
        st.write(chain.run(input_text))
