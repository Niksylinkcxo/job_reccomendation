import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the Excel files
jobs_file_path = 'C:\\Users\\Nikhil Parsarwar\\Downloads\\jobs\\jobs.xlsx'
user_details_file_path = 'C:\\Users\\Nikhil Parsarwar\\Downloads\\jobs\\User Details.xlsx'

# Load the sheets
jobs_df = pd.read_excel(jobs_file_path)
user_details_df = pd.read_excel(user_details_file_path)

# Handle missing values in the user details dataframe
user_details_df.fillna('', inplace=True)

# Function to clean job descriptions
def clean_job_description(text):
    if isinstance(text, str):
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        return text.strip()
    return ''

# Clean the job descriptions
jobs_df['cleaned_description'] = jobs_df['job_description'].apply(clean_job_description)

# Function to clean and standardize the work experience field
def clean_work_experience(text):
    if isinstance(text, str):
        text = text.strip()
        if re.match(r'\d+\s*[-+]\s*\d+\\s*years', text):
            return text
        elif re.match(r'\d+\s*years', text):
            return text
    return 'Unknown'

# Clean the work experience field
user_details_df['cleaned_work_ex'] = user_details_df['work-ex'].apply(clean_work_experience)

# Combine relevant user details into a single string for feature extraction
user_details_df['combined_info'] = user_details_df.apply(
    lambda row: ' '.join([row['position'], row['cleaned_work_ex'], row['domain']]), axis=1
)

# Feature extraction using TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
job_descriptions_tfidf = vectorizer.fit_transform(jobs_df['cleaned_description'])
user_profiles_tfidf = vectorizer.transform(user_details_df['combined_info'])

# Compute similarity scores using cosine similarity
similarity_scores = cosine_similarity(user_profiles_tfidf, job_descriptions_tfidf)

# Function to get top N job recommendations for a given user index
def get_top_recommendations(user_index, top_n=5):
    scores = similarity_scores[user_index]
    top_indices = scores.argsort()[-top_n:][::-1]
    recommendations = pd.DataFrame({
        'index': top_indices,
        'similarity_score': scores[top_indices],
        'cleaned_description': jobs_df.iloc[top_indices]['cleaned_description'].values,
        'job_title': jobs_df.iloc[top_indices]['job_title'].values,
        'company_name': jobs_df.iloc[top_indices]['company_name'].values
    })
    return recommendations

# Compute similarity scores among job descriptions
job_similarity_scores = cosine_similarity(job_descriptions_tfidf, job_descriptions_tfidf)

# Function to get top N similar jobs for a given job index
def get_similar_jobs(job_index, top_n=5):
    scores = job_similarity_scores[job_index]
    top_indices = scores.argsort()[-top_n-1:-1][::-1]  # Exclude the job itself
    similar_jobs = pd.DataFrame({
        'index': top_indices,
        'similarity_score': scores[top_indices],
        'cleaned_description': jobs_df.iloc[top_indices]['cleaned_description'].values,
        'job_title': jobs_df.iloc[top_indices]['job_title'].values,
        'company_name': jobs_df.iloc[top_indices]['company_name'].values
    })
    return similar_jobs

# Streamlit UI
st.title("Job Recommendation System")

# First Page: User Input and Job Recommendations
st.header("Enter your profile information")

user_position = st.text_input("Current Position")
user_work_experience = st.text_input("Work Experience in years")
user_domain = st.text_input("Domain of Work")

# Button to trigger recommendations
if st.button("Get Job Recommendations"):
    # Combine user input into a single string for feature extraction
    user_info = f"{user_position} {user_work_experience} {user_domain}"
    user_info_tfidf = vectorizer.transform([user_info])

    # Compute similarity scores for the user input
    user_similarity_scores = cosine_similarity(user_info_tfidf, job_descriptions_tfidf)
    
    # Get top 5 job recommendations
    top_indices = user_similarity_scores[0].argsort()[-5:][::-1]
    recommendations = jobs_df.iloc[top_indices].copy()
    recommendations['similarity_score'] = user_similarity_scores[0, top_indices]

    st.header("Top Job Recommendations")
    recommendations_df = recommendations[['job_title', 'company_name', 'similarity_score', 'cleaned_description']].reset_index(drop=True)
    st.dataframe(recommendations_df)

    for i, row in recommendations_df.iterrows():
        with st.expander(f"{row['job_title']} - {row['company_name']}"):
            st.write(f"Description: {row['cleaned_description']}")
            st.write(f"Similarity Score: {row['similarity_score']:.2f}")
            
            # Get similar jobs for this recommendation
            similar_jobs_df = get_similar_jobs(row.name, top_n=5)[['job_title', 'company_name', 'similarity_score', 'cleaned_description']].reset_index(drop=True)
            st.write("Top 5 Similar Jobs:")
            st.dataframe(similar_jobs_df)
            
            for j, similar_row in similar_jobs_df.iterrows():
                st.write(f"**Similar Job: {similar_row['job_title']} - {similar_row['company_name']}**")
                st.write(f"Description: {similar_row['cleaned_description']}")
                st.write(f"Similarity Score: {similar_row['similarity_score']:.2f}")

# Similar Jobs Page
if "selected_job_index" in st.session_state:
    selected_job_index = st.session_state["selected_job_index"]

    st.header("Similar Jobs to the Selected Job")
    
    similar_jobs_df = get_similar_jobs(selected_job_index, top_n=5)[['job_title', 'company_name', 'similarity_score', 'cleaned_description']].reset_index(drop=True)
    st.dataframe(similar_jobs_df)
    
    for i, row in similar_jobs_df.iterrows():
        with st.expander(f"{row['job_title']} - {row['company_name']}"):
            st.write(f"Description: {row['cleaned_description']}")
            st.write(f"Similarity Score: {row['similarity_score']:.2f}")
            
            # Get similar jobs for this job
            similar_jobs_df_inner = get_similar_jobs(row.name, top_n=5)[['job_title', 'company_name', 'similarity_score', 'cleaned_description']].reset_index(drop=True)
            st.write("Top 5 Similar Jobs:")
            st.dataframe(similar_jobs_df_inner)
            
            for j, similar_row in similar_jobs_df_inner.iterrows():
                st.write(f"**Similar Job: {similar_row['job_title']} - {similar_row['company_name']}**")
                st.write(f"Description: {similar_row['cleaned_description']}")
                st.write(f"Similarity Score: {similar_row['similarity_score']:.2f}")
