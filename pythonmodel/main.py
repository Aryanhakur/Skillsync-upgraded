import pandas as pd
import json
import spacy
import requests
import pdfplumber
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from bs4 import BeautifulSoup
import ast
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    nlp = spacy.load("output2")
except Exception as e:
    print(f"Error loading spaCy model: {e}")
    # Create a fallback model
    nlp = None

# Common tech skills for fallback
COMMON_TECH_SKILLS = [
    "python", "java", "javascript", "html", "css", "react", "angular", "vue",
    "node.js", "express", "django", "flask", "spring", "hibernate", "sql",
    "mysql", "postgresql", "mongodb", "nosql", "aws", "azure", "gcp",
    "docker", "kubernetes", "jenkins", "git", "github", "gitlab", "ci/cd",
    "agile", "scrum", "jira", "confluence", "rest api", "graphql", "microservices",
    "machine learning", "artificial intelligence", "data science", "big data",
    "hadoop", "spark", "tensorflow", "pytorch", "keras", "nlp", "computer vision",
    "devops", "sre", "cloud computing", "serverless", "linux", "unix", "bash",
    "powershell", "c#", "c++", "ruby", "php", "swift", "kotlin", "go", "rust",
    "typescript", "jquery", "bootstrap", "sass", "less", "webpack", "babel",
    "redux", "vuex", "next.js", "nuxt.js", "gatsby", "graphql", "rest",
    "oauth", "jwt", "authentication", "authorization", "security", "encryption",
    "testing", "unit testing", "integration testing", "e2e testing", "jest",
    "mocha", "chai", "selenium", "cypress", "puppeteer", "responsive design",
    "mobile development", "ios", "android", "react native", "flutter", "xamarin"
]

# Takes file path as input and returns the text from the resume
def extract_pdf_text(path):
    text=''
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text+=page.extract_text()+'\n'
    return text

# Takes raw text extracted from the resume and returns a comma separated string of skills
def extract_skills(text):
    # If spaCy model is available, use it
    if nlp is not None:
        try:
            skills = []
            docs = nlp(text)
            for doc in docs.ents:
                skills.append(str.lower(doc.text))
            skills = list(set(skills))
            if skills:  # If skills were found
                return ', '.join(skills)
        except Exception as e:
            print(f"Error using spaCy model: {e}")
            # Fall through to the fallback method

    # Fallback method: Look for common tech skills in the text
    text_lower = text.lower()
    found_skills = []

    for skill in COMMON_TECH_SKILLS:
        if skill in text_lower:
            found_skills.append(skill)

    # If no skills found, return some default skills to avoid empty results
    if not found_skills:
        found_skills = ["python", "javascript", "html", "css", "sql"]

    return ', '.join(found_skills)

# To one_hot encode the skills to be used for association rule mining
def one_hot_encode_skills(skills):
    if isinstance(skills, str):  # If input is a single string
        skills = pd.Series([skills])  # Convert to Series

    # Split each entry by ", " and expand into separate rows
    skills_split = skills.str.get_dummies(sep=", ")
    return skills_split

# Takes user skills and job_listings as input and returns a dataframe with jobs sorted by their similarity score
def recommended_jobs(user_skills,job_listings):
    all_skills = [user_skills] + job_listings["Skills"].tolist()  # Combine user skills & job skills
    vectorizer = TfidfVectorizer()
    skill_vectors = vectorizer.fit_transform(all_skills)  # Convert text to numerical vectors
    cosine_sim = cosine_similarity(skill_vectors[0:1], skill_vectors[1:])  # Compare user skills with jobs
    job_listings["Similarity Score"] = cosine_sim[0]
    recommended_jobs = job_listings.sort_values(by="Similarity Score", ascending=False)
    return recommended_jobs

# Suggests new skills to the user based on their existing skills and skills extracted from the job descriptions
def recommended_skills(user_skills,job_listings,Skill_number=None):
    user_skills=user_skills.split(", ")
    job_listings['Skills_list']=job_listings['Skills'].apply(lambda x: x.split(", "))
    all_job_skills = [skill for skills_list in job_listings["Skills_list"] for skill in skills_list]
    skill_counts = Counter(all_job_skills)
    missing_skills = set(skill_counts.keys()) - set(user_skills)
    recommended_skills = sorted(missing_skills, key=lambda skill: skill_counts[skill], reverse=True)
    # 5️ Display Top Suggested Skills
    if Skill_number is None:
        return recommended_skills[:10]
    else:
        return recommended_skills[:Skill_number]

# Gets certifications names and links for Certifications based on the skills entered by the user
def get_certifications(specialization):
    Courses = []
    limit_per_skill = 3  # Limit to exactly 3 certifications per skill

    # Check if specialization contains multiple skills (comma-separated)
    if ',' in specialization:
        # Split the skills and use all of them (up to 10) for better results
        skills = [skill.strip() for skill in specialization.split(',') if skill.strip()][:10]
        print(f"Searching for certifications for multiple skills: {skills}")

        # Search for each skill and combine results, limiting to exactly 3 per skill
        for skill in skills:
            print(f"Processing skill: {skill}")
            skill_courses = search_coursera_for_skill(skill)
            print(f"Found {len(skill_courses)} courses for skill: {skill}")

            # Limit to exactly 3 courses per skill
            limited_courses = skill_courses[:limit_per_skill]
            print(f"Using {len(limited_courses)} courses for skill: {skill}")

            # Add skill name as metadata to each course
            for course in limited_courses:
                # Add the skill as the third element in the course tuple
                if len(course) == 2:
                    course_with_skill = [course[0], course[1], skill]
                    Courses.append(course_with_skill)
                else:
                    Courses.append(course)

        # Remove duplicates while preserving order
        unique_courses = []
        seen_titles = set()
        for course in Courses:
            if course[0] not in seen_titles:
                seen_titles.add(course[0])
                unique_courses.append(course)

        print(f"Total unique courses found: {len(unique_courses)}")
        return unique_courses  # Return all unique courses (should be up to 3 per skill)
    else:
        # Single skill/specialization search
        print(f"Searching for certifications for single skill: {specialization}")
        skill_courses = search_coursera_for_skill(specialization)

        # Limit to exactly 3 courses
        limited_courses = skill_courses[:limit_per_skill]
        print(f"Returning {len(limited_courses)} courses for single skill: {specialization}")

        return limited_courses

# Helper function to search Coursera for a specific skill
def search_coursera_for_skill(skill):
    print(f"Searching Coursera for: {skill}")
    skill_courses = []

    if not skill or len(skill.strip()) == 0:
        print("Empty skill provided, skipping search")
        return []

    # Construct the URL with the skill as the query parameter
    url = f"https://www.coursera.org/search?query={skill}&sortBy=BEST_MATCH"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "max-age=0"
    }  # More complete headers to avoid being blocked

    try:
        print(f"Making request to: {url}")
        response = requests.get(url, headers=headers, timeout=10)  # Add timeout
        response.raise_for_status()  # Raise exception for HTTP errors

        # Parse the HTML content
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract course links - try different class patterns as Coursera might change them
        links = soup.find_all('a', class_='cds-119 cds-113 cds-115 cds-CommonCard-titleLink css-vflzcf cds-142')

        # If no links found with the first pattern, try alternative patterns
        if not links:
            links = soup.find_all('a', class_=lambda c: c and 'CommonCard-titleLink' in c)

        # If still no links, try a more generic approach
        if not links:
            links = soup.find_all('a', href=lambda h: h and '/learn/' in h)

        print(f"Found {len(links)} potential course links")

        for link in links:
            # Try to extract course title and URL
            title = None
            if link.h3 and link.h3.string:
                title = link.h3.string
            elif link.get('aria-label'):
                title = link.get('aria-label')
            elif link.text:
                title = link.text.strip()

            if title and link.get('href'):
                href = link['href']
                # Ensure URL is absolute
                if not href.startswith('http'):
                    href = "https://www.coursera.org" + href

                course = [title, href]
                skill_courses.append(course)

        print(f"Successfully extracted {len(skill_courses)} courses for skill: {skill}")
        return skill_courses

    except requests.exceptions.RequestException as e:
        print(f"Request error searching Coursera for {skill}: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error searching Coursera for {skill}: {e}")
        return []

# Extracts the links for jobs to apply from the recommended jobs DataFrame
def get_job_links(recommended_jobs):
    job_links_df=pd.DataFrame(columns=['jobProvider','url'])
    for i in recommended_jobs['jobProviders']:
        li=ast.literal_eval(i)
        row=pd.DataFrame(data=li[:2],columns=['jobProvider','url'])
        job_links_df=pd.concat([job_links_df,row],ignore_index=True)
    return job_links_df


# Fetches job listings
def fetch_paginated_jobs(api_url, params, headers=None, max_pages=5):
    all_jobs = []
    next_page = ""  # Initialize nextPage as empty

    for _ in range(max_pages):  # Limit number of pages to fetch
        if next_page:  # Add nextPage token only if it exists
            params["nextPage"] = next_page
        try:
            response = requests.get(api_url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()

            jobs = data.get("jobs", [])  # Extract job data
            all_jobs.extend(jobs)

            # Get nextPage token for next request
            next_page = data.get("nextPage")  # Adjust based on API response

            if not next_page:  # Stop if no more pages
                break

        except requests.exceptions.RequestException as e:
            print(f"Error fetching jobs: {e}")
            break

    return all_jobs

# Example Usage
api_url = "https://jobs-api14.p.rapidapi.com/v2/list"

params = {"query":"",
            "location":"India",
            "autoTranslateLocation":"true",
            "remoteOnly":"false",
            "employmentTypes":"fulltime;parttime;intern;contractor"}

headers = {
	"x-rapidapi-key": os.environ.get('RAPIDAPI_KEY', '726bb03d58mshefcca0953ad0d7ap1b6e63jsn054dc3c55e8a'),
	"x-rapidapi-host": os.environ.get('RAPIDAPI_HOST', 'jobs-api14.p.rapidapi.com')
}

profile=None
Location=None
remote=None


if profile:
    params['query']=profile
if Location:
    params['location']=Location
if remote:
    params['remoteOnly']=remote



jobs_df=pd.read_csv("Job_listings.csv")
listings=pd.DataFrame(jobs_df)
#  listings.to_csv('Job_listings.csv',index=False)
listings['Skills']=listings['description'].apply(lambda x:extract_skills(x))
print(listings)


def main():
    text=extract_pdf_text("specialised2vansh.pdf")
    Skills_text=extract_skills(text)
    print('The user has the Following Skills: \n')
    for i in Skills_text.split(', '):
        print(i)


    print('\nThe Following Jobs match the user profile and are sorted in the descending order of similarity: ')
    rec_jobs=recommended_jobs(Skills_text,listings)
    print(rec_jobs[['id','title','Similarity Score']])

    print('\nThe following Skills can be added to the user resume based\non the extracted Skills from the user resume and Job listings data received:\n')
    rec_skills=recommended_skills(Skills_text,listings)
    for i in rec_skills:
        print(i)

    Courses=get_certifications(rec_jobs['title'])
    Courses=pd.DataFrame(data=Courses,columns=['Course','Link'])
    print('\n The user Can get Following Certifications\n')
    print(Courses)

    print("\nHere are the links to the Job listings\n")
    jobs_links=get_job_links(rec_jobs)
    print(jobs_links)

if __name__ == "__main__":
    main()

