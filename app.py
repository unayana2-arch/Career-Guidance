

# Define the unique values for categorical features from your dataset
# These should ideally come from the training data to ensure consistency
# For demonstration, I'm inferring some based on previous outputs.
# You might need to adjust these lists to reflect all unique values in your actual training data.
INTERESTS = ['Maths', 'Arts', 'Science', 'Business', 'Medicine', 'Sports', 'Technology', 'Design']
STRENGTHS = ['Communication', 'Analysis', 'Problem Solving', 'Logical Thinking', 'Leadership', 'Creativity']
SKILLS = ['Java', 'Biology', 'Drawing', 'Physics', 'Public Speaking', 'Marketing', 'Statistics', 'Coding']
PERSONALITY = ['Introvert', 'Extrovert', 'Ambivert']
GOALS = ['Start a business', 'Develop new skills', 'Higher studies', 'Get a good job']
BUDGET = ['High', 'Medium', 'Low']
RECOMMENDED_COURSE = ['Business Management', 'Graphic Design', 'Data Science Intro', 'Biology 101', 'Web Development', 'Marketing Strategies']
SCHOLARSHIP_OPTION = ['National Merit Scholarship', 'Need-based Aid', 'STEM Scholarship', 'Arts Talent Scholarship']

st.title('Career Guidance Predictor')
st.write('Enter student details to get a career suggestion.')

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
