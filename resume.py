import os
import re
import PyPDF2
import docx2txt
import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, render_template
# Load NLP Model
nlp = spacy.load('en_core_web_sm')
# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text
# Function to extract text from DOCX
def extract_text_from_docx(docx_path):
    return docx2txt.process(docx_path)
# Function to clean text
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    return text
# Function to extract skills
def extract_skills(text):
    skills = []
    doc = nlp(text)
    for token in doc:
        if token.pos_ == "NOUN" or token.pos_ == "PROPN":
            skills.append(token.text)
    return set(skills)
# Function to match resume with job description
def match_resume_with_job(resume_text, job_description):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_description])
    similarity_score = cosine_similarity(vectors[0], vectors[1])
    return similarity_score[0][0]
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
        score = match_resume_with_job(cleaned_resume_text, clean_text(job_description))
        skills = extract_skills(cleaned_resume_text)
        
        return render_template('result.html', score=score, skills=skills)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
