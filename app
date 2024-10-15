from flask import Flask, request, jsonify
import pandas as pd
import logging
import random
from pymongo import MongoClient
import google.generativeai as genai

app = Flask(__name__)




# MongoDB connection setup for a specific database
def connect_mongodb(uri, database_name):
    client = MongoClient(uri)
    return client[database_name]

# Load job data from MongoDB collection in the job database
def load_job_data_from_mongo(job_db):
    collection = job_db["job"]  # Replace with your actual job collection name
    job_data = list(collection.find())
    print(f"Loaded {len(job_data)} jobs from MongoDB")
    if len(job_data) == 0:
        print("No job data found in MongoDB.")
    return pd.DataFrame(job_data)

# Load user details data from MongoDB collection in the user database
def load_user_data_from_mongo(user_db):
    collection = user_db["user"]  # Replace with your actual user collection name
    user_data = list(collection.find())
    print(f"Loaded {len(user_data)} users from MongoDB")
    return pd.DataFrame(user_data)

import google.generativeai as genai

# Set up Google Generative AI API
api_list = ["AIzaSyDbkKLO5VmTaMXu4MUgEpF5QpXB3J2iNKQ"]
api_key = random.choice(api_list)
print(f"Using API Key: {api_key}")

genai.configure(api_key=api_key)

mongo_uri = "mongodb+srv://linkcxoadmin:PxDpyQH3p1ie1nq@prod-cluster.rtezr.mongodb.net/"  # Replace with your MongoDB URI
job_database_name = "api-job-service"  # Replace with your job database name
user_database_name = "api-user-service"  # Replace with your user database name


# Connect to both MongoDB databases
job_db = connect_mongodb(mongo_uri, job_database_name)
user_db = connect_mongodb(mongo_uri, user_database_name)

# Load job and user data from respective databases
job_list = load_job_data_from_mongo(job_db)
user_details = load_user_data_from_mongo(user_db)

# Filter only open jobs
job_list = job_list[job_list['jobStatus'] == 'OPEN']


# Function to generate a response using Gemini model
def get_gemini_response(user_text, job_strings, prompt):
    responses = []
    for job_string in job_strings:
        try:
            response = model.generate_content([user_text, job_string, prompt], generation_config=genai.GenerationConfig(
                temperature=0.4
            ))
            responses.append(response.text)
        except Exception as e:
            logging.error(f"Error generating Gemini response: {e}")
            responses.append("Error generating response.")
    return responses




genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

cache = {}

def get_job_recommendations_paginated(designation, organization, industries=None, location=None, experience=None, skills=None, page=None, limit=None):
    user_text = f"{designation} {organization}"
    
    if industries:
        user_text += f" {industries}"
    if location:
        user_text += f" {location}"
    if experience:
        user_text += f" {experience}"
    if skills:
        user_text += f" {skills}"
    
    logging.info(f"User Input: {user_text}")
    print(f"User Input for job recommendations: {user_text}")
    
    # Check if job list DataFrame is loaded properly
    print(f"Job List Columns: {job_list.columns}")
    print(f"Job List Preview before filtering: {job_list.head()}")

    # If job_list is empty, stop further processing
    if job_list.empty:
        print("Job list is empty. Cannot proceed with recommendations.")
        return []

    # Pagination logic
    start = (page - 1) * limit
    end = start + limit
    paginated_job_list = job_list[start:end]

        # Generate job strings for the Gemini model
    job_strings = paginated_job_list.apply(lambda row: f"ID: {row['_id']} Title: {row['title']} Details: {row['description']} Skills: {row['skills']}", axis=1).tolist()
    print(f"Generated job strings for model input: {job_strings[:5]}")  # Preview only first 5 jobs

    # Set up a prompt for Gemini model
    prompt = (
        "Based on the user's job profile and skills, identify and recommend the most relevant jobs from the provided list. "
        "Rank the jobs in order of relevance."
    )
    
    responses = get_gemini_response(user_text, job_strings, prompt)
    
    scores_with_indices = []
    for idx, responses in enumerate(responses):
        try:
            relevance_score = random.uniform(0, 100)  # Placeholder for a real scoring mechanism
            scores_with_indices.append((relevance_score, idx))
            print(f"Generated relevance score {relevance_score:.2f} for job at index {idx}")
        except Exception as e:
            logging.error(f"Error processing Gemini response: {e}")
            continue
    
    scores_with_indices.sort(reverse=True, key=lambda x: x[0])
    print(f"Sorted job scores: {scores_with_indices}")

    all_indices = [idx for _, idx in scores_with_indices]
    recommended_jobs = paginated_job_list.iloc[all_indices].copy()
    
    recommended_jobs['similarity_score'] = [f"{score:.2f}" for score, _ in scores_with_indices]
    print(f"Recommended jobs with similarity scores: {recommended_jobs.head()}")

    # Convert _id to id and format output with all necessary fields
    recommended_jobs['id'] = recommended_jobs['_id']
    
    # Selecting and returning all required fields
    return recommended_jobs[[
       'id', 'title', 'company', 'companyId', 'hideCompany', 'industries', 'functions', 
    'salary', 'hideSalary', 'location', 'jobType', 'experience', 'hiringFor', 
    'qualification', 'skills', 'description', 'similarity_score', 'jobStatus', 
    'workplaceType', 'currency', 'noOfApplicants', 'createdAt', 'updatedAt', 
    'attachment', 'authorType', 'authorId', 'createdBy', 'contactEmailId'
    ]].reset_index(drop=True).to_dict(orient='records')


@app.route('/v1/get_job_recommendations', methods=['GET'])
def recommendations():
    # Get query parameters
    designation = request.args.get('designation')
    organization = request.args.get('organization')
    industry = request.args.get('industry')
    location = request.args.get('location')
    totalExperience = request.args.get('totalExperience')
    skills = request.args.get('skills')
    page = int(request.args.get('page', 1))  # Default to page 1 if not provided
    limit = int(request.args.get('limit', 5))  # Default to limit 10 if not provided

    # Validate required parameters
    if not designation or not organization:
        return jsonify({'error': 'Designation and Organization are required'}), 400

    # Call the job recommendation function with pagination
    recommendations = get_job_recommendations_paginated(designation, organization, industry, location, totalExperience, skills, page, limit)
   
  

    

    # Prepare the response
    response = {
        'page': page,
        'limit': limit,
        'total_recommendations': len(job_list),
        'total_pages': (len(job_list) + limit - 1) // limit,  # Calculate total pages
        'recommendations': recommendations
    }

    return jsonify(response)



if __name__ == '__main__':
    app.run(debug=True)
