import os
import re
import PyPDF2
import docx2txt
import spacy
import pandas as pd
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, render_template

# Load NLP Model
nlp = spacy.load('en_core_web_sm')

# Function to extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text + "\n"
    return text if text.strip() else "No text found in PDF."

# Function to extract text from DOCX
def extract_text_from_docx(file):
    return docx2txt.process(file)

# Function to clean text
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove numbers and special characters but keep spaces
    text = text.lower().strip()
    return text

# Function to extract relevant skills dynamically using fuzzy matching
def extract_skills_from_resume(resume_text, job_description):
    resume_doc = nlp(resume_text)
    job_doc = nlp(job_description)
    
    job_keywords = {token.text.lower() for token in job_doc if token.is_alpha and not token.is_stop}
    resume_keywords = {token.text.lower() for token in resume_doc if token.is_alpha and not token.is_stop}
    
    matched_skills = set()
    
    for job_skill in job_keywords:
        for resume_skill in resume_keywords:
            if fuzz.ratio(job_skill, resume_skill) > 80:  # If similarity > 80%, consider it a match
                matched_skills.add(resume_skill)
    
    return list(matched_skills)

# Function to match resume with job description
def match_resume_with_job(resume_text, job_description):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_description])
    
    print("TF-IDF Features:", vectorizer.get_feature_names_out())  # Debugging line
    print("Resume Vector:", vectors[0].toarray())  # Debugging line
    print("Job Desc Vector:", vectors[1].toarray())  # Debugging line

    similarity_score = cosine_similarity(vectors[0], vectors[1])
    return round(float(similarity_score[0][0]) * 100, 2)

# Flask App Setup
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        resume_file = request.files['resume']
        job_description = request.form['job_description']

        if resume_file.filename.endswith('.pdf'):
            resume_text = extract_text_from_pdf(resume_file)
        elif resume_file.filename.endswith('.docx'):
            resume_text = extract_text_from_docx(resume_file)
        else:
            return "Unsupported file format", 400

        cleaned_resume_text = clean_text(resume_text)
        cleaned_job_desc = clean_text(job_description)
        
        print("Cleaned Resume Text:", cleaned_resume_text)  # Debugging line
        print("Cleaned Job Description:", cleaned_job_desc)  # Debugging line

        score = match_resume_with_job(cleaned_resume_text, cleaned_job_desc)
        skills = extract_skills_from_resume(cleaned_resume_text, cleaned_job_desc)

        return render_template('result.html', score=score, skills=skills)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)

