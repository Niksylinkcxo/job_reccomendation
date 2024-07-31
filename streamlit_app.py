import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the Excel files
jobs_file_path = './job_data.csv'
user_details_file_path = './User_Details.xlsx'

# Load the sheets
jobs_df = pd.read_csv(jobs_file_path, encoding='ISO-8859-1')
user_details_df = pd.read_excel(user_details_file_path)

# Handle missing values in the user details dataframe
user_details_df.fillna('', inplace=True)

# Function to clean job descriptions
def clean_job_description(text):
    if isinstance(text, str):
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        return text.strip()
    return ''

# Filter out closed jobs
open_jobs_df = jobs_df[jobs_df['status'].str.lower() == 'open']

# Clean the job descriptions in the open jobs dataframe
open_jobs_df['cleaned_description'] = open_jobs_df['job_description'].apply(clean_job_description)

# Function to clean and standardize the work experience field
def clean_work_experience(text):
    if isinstance(text, str):
        text = text.strip()
        if re.match(r'\d+\s*[-+]\s*\d+\s*years', text):
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
job_descriptions_tfidf = vectorizer.fit_transform(open_jobs_df['cleaned_description'])
user_profiles_tfidf = vectorizer.transform(user_details_df['combined_info'])

# Compute similarity scores using cosine similarity
similarity_scores = cosine_similarity(user_profiles_tfidf, job_descriptions_tfidf)

# Function to get top N job recommendations for a given user index
def get_top_recommendations(user_index, top_n=5):
    scores = similarity_scores[user_index]
    top_indices = scores.argsort()[-top_n:][::-1]
    recommendations = pd.DataFrame({
        'index': top_indices,
        'cleaned_description': open_jobs_df.iloc[top_indices]['cleaned_description'].values,
        'job_title': open_jobs_df.iloc[top_indices]['job_title'].values,
        'company_name': open_jobs_df.iloc[top_indices]['company_name'].values
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
        'cleaned_description': open_jobs_df.iloc[top_indices]['cleaned_description'].values,
        'job_title': open_jobs_df.iloc[top_indices]['job_title'].values,
        'company_name': open_jobs_df.iloc[top_indices]['company_name'].values
    })
    return similar_jobs

# Recursive function using while loop to get similar jobs up to a specified depth without duplicates
def get_recursive_similar_jobs(job_index, depth=5, top_n=5):
    stack = [(job_index, depth)]
    visited = set()
    recursive_results = []

    while stack:
        current_job_index, current_depth = stack.pop()
        if current_job_index in visited or current_depth == 0:
            continue
        visited.add(current_job_index)
        
        similar_jobs = get_similar_jobs(current_job_index, top_n)
        for _, row in similar_jobs.iterrows():
            if row['index'] not in visited:
                job_details = {
                    'job_index': row['index'],
                    'job_title': row['job_title'],
                    'company_name': row['company_name'],
                    'cleaned_description': row['cleaned_description'],
                    'similar_jobs': []
                }
                recursive_results.append(job_details)
                stack.append((row['index'], current_depth - 1))
    
    return recursive_results

# Streamlit UI
st.title("Job Recommendation System")

# First Page: User Input and Job Recommendations
st.header("Enter your profile information")

user_position = st.text_input("Current Position")
user_work_experience = st.text_input("Work Experience in years")
user_domain = st.text_input("Your Industry")
user_summary = st.text_input("Write your Key Skills or Summary")

# Button to trigger recommendations
if st.button("Get Job Recommendations"):
    # Combine user input into a single string for feature extraction
    user_info = f"{user_position} {user_work_experience} {user_domain} {user_summary}"
    user_info_tfidf = vectorizer.transform([user_info])

    # Compute similarity scores for the user input against open job descriptions
    user_similarity_scores = cosine_similarity(user_info_tfidf, job_descriptions_tfidf)
    
    # Get top 5 job recommendations
    top_indices = user_similarity_scores[0].argsort()[-5:][::-1]
    recommendations = open_jobs_df.iloc[top_indices].copy()

    st.header("Top Job Recommendations")
    recommendations_df = recommendations[['job_title', 'company_name', 'cleaned_description']].reset_index(drop=True)
    st.dataframe(recommendations_df)

    for i, row in recommendations_df.iterrows():
        with st.expander(f"{row['job_title']} - {row['company_name']}"):
            st.write(f"Description: {row['cleaned_description']}")
            
            # Get similar jobs for this recommendation recursively
            recursive_similar_jobs = get_recursive_similar_jobs(row.name, depth=5, top_n=5)
            for similar_job in recursive_similar_jobs:
                st.write(f"**Similar Job: {similar_job['job_title']} - {similar_job['company_name']}**")
                st.write(f"Description: {similar_job['cleaned_description']}")

                for inner_similar_job in similar_job['similar_jobs']:
                    st.write(f"***Inner Similar Job: {inner_similar_job['job_title']} - {inner_similar_job['company_name']}***")
                    st.write(f"Description: {inner_similar_job['cleaned_description']}")
