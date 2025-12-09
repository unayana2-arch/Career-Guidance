
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- WARNING: This approach is NOT recommended for production Streamlit apps ---
# The model will be retrained every time the app starts or refreshes, leading to poor performance.
# For production, train the model once, save it (e.g., with joblib), and load the saved model.
# --- End WARNING ---

st.title('Career Guidance Predictor')
st.write('Enter student details to get a career suggestion.')

@st.cache_resource # Use Streamlit's caching to train the model only once per app session
def train_model():
    # 1. Load the dataset (replace with your actual path if different)
    try:
        df = pd.read_csv('/content/student_career_guidance_dataset_1000.csv')
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure 'student_career_guidance_dataset_1000.csv' is in the same directory.")
        st.stop()

    # 2. Define input (ip) and output (op)
    ip = df.drop(columns=['Student_ID', 'Suggested_Career'])
    op = df['Suggested_Career']

    # 3. Split the data
    ip_train, _, op_train, _ = train_test_split(ip, op, random_state=20)

    # 4. Identify categorical columns
    categorical_cols = ip_train.select_dtypes(include=['object']).columns

    # 5. Create a preprocessor for one-hot encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'
    )

    # 6. Create and train the model pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=20))
    ])
    model_pipeline.fit(ip_train, op_train)
    return model_pipeline, ip_train # Return ip_train to get column names for selectbox options

# Train the model (cached)
model_pipeline, ip_train_for_options = train_model()

# Define the unique values for categorical features from your training data
# This ensures consistency with what the model expects after one-hot encoding.
# We extract unique values from the training data's categorical columns.
# Fallback to hardcoded lists if issue during extraction.

def get_unique_values(column_name, fallback_list):
    if column_name in ip_train_for_options.columns:
        return sorted(ip_train_for_options[column_name].unique().tolist())
    return fallback_list

INTERESTS = get_unique_values('Interests', ['Maths', 'Arts', 'Science', 'Business', 'Medicine', 'Sports', 'Technology', 'Design'])
STRENGTHS = get_unique_values('Strengths', ['Communication', 'Analysis', 'Problem Solving', 'Logical Thinking', 'Leadership', 'Creativity'])
SKILLS = get_unique_values('Skills', ['Java', 'Biology', 'Drawing', 'Physics', 'Public Speaking', 'Marketing', 'Statistics', 'Coding'])
PERSONALITY = get_unique_values('Personality', ['Introvert', 'Extrovert', 'Ambivert'])
GOALS = get_unique_values('Goals', ['Start a business', 'Develop new skills', 'Higher studies', 'Get a good job'])
BUDGET = get_unique_values('Budget', ['High', 'Medium', 'Low'])
RECOMMENDED_COURSE = get_unique_values('Recommended_Course', ['Business Management', 'Graphic Design', 'Data Science Intro', 'Biology 101', 'Web Development', 'Marketing Strategies'])
SCHOLARSHIP_OPTION = get_unique_values('Scholarship_Option', ['National Merit Scholarship', 'Need-based Aid', 'STEM Scholarship', 'Arts Talent Scholarship'])

# Input fields for features
marks = st.slider('Marks (0-100)', 0, 100, 75)
interests = st.selectbox('Interests', INTERESTS)
strengths = st.selectbox('Strengths', STRENGTHS)
skills = st.selectbox('Skills', SKILLS)
personality = st.selectbox('Personality', PERSONALITY)
goals = st.selectbox('Goals', GOALS)
budget = st.selectbox('Budget', BUDGET)
recommended_course = st.selectbox('Recommended Course', RECOMMENDED_COURSE)
scholarship_option = st.selectbox('Scholarship Option', SCHOLARSHIP_OPTION)

if st.button('Predict Suggested Career'):
    # Create a DataFrame from the inputs
    input_data = pd.DataFrame([{
        'Marks': marks,
        'Interests': interests,
        'Strengths': strengths,
        'Skills': skills,
        'Personality': personality,
        'Goals': goals,
        'Budget': budget,
        'Recommended_Course': recommended_course,
        'Scholarship_Option': scholarship_option
    }])

    # Make prediction
    prediction = model_pipeline.predict(input_data)

    st.success(f'The suggested career is: **{prediction[0]}**')
