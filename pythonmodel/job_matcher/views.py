from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from django.conf import settings
import os
import sys
import pandas as pd
import requests
from .serializers import JobSerializer, CertificationSerializer, JobLinkSerializer

# Add the project root to the Python path to import main.py
sys.path.append(settings.BASE_DIR)

try:
    from main import (
        extract_pdf_text,
        extract_skills,
        recommended_jobs,
        recommended_skills,
        get_certifications,
        get_job_links
    )
except ImportError as e:
    print(f"Error importing functions from main.py: {e}")
    # Define fallback functions if imports fail
    def extract_pdf_text(path):
        try:
            import pdfplumber
            text = ""
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + '\n'
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""

    def extract_skills(text):
        # Simple fallback implementation
        common_skills = [
            "python", "java", "javascript", "html", "css", "react", "angular", "vue",
            "node.js", "express", "django", "flask", "spring", "hibernate", "sql",
            "mysql", "postgresql", "mongodb", "nosql", "aws", "azure", "gcp",
            "docker", "kubernetes", "jenkins", "git", "github", "gitlab", "ci/cd",
            "agile", "scrum", "jira", "confluence", "rest api", "graphql", "microservices"
        ]

        found_skills = []
        text_lower = text.lower()

        for skill in common_skills:
            if skill in text_lower:
                found_skills.append(skill)

        if not found_skills:
            found_skills = ["python", "javascript", "html", "css", "sql"]

        return ", ".join(found_skills)

    def recommended_jobs(user_skills, job_listings):
        return job_listings.head(5)

    def recommended_skills(user_skills, job_listings, skill_number=None):
        return ["python", "javascript", "react", "node.js", "sql"]

    def get_certifications(specialization):
        return [["Python Programming", "https://www.coursera.org/python"],
                ["Web Development", "https://www.coursera.org/webdev"]]

    def get_job_links(recommended_jobs):
        return pd.DataFrame(columns=['jobProvider', 'url'])


class ExtractSkillsView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        if 'resume' not in request.FILES:
            return Response({"error": "No resume file provided"}, status=status.HTTP_400_BAD_REQUEST)

        resume_file = request.FILES['resume']

        # Validate file type
        if not resume_file.name.lower().endswith('.pdf'):
            return Response({"error": "Only PDF files are supported"}, status=status.HTTP_400_BAD_REQUEST)

        # Create media directory if it doesn't exist
        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)

        # Save the file temporarily
        file_path = os.path.join(settings.MEDIA_ROOT, resume_file.name)
        try:
            with open(file_path, 'wb+') as destination:
                for chunk in resume_file.chunks():
                    destination.write(chunk)
        except Exception as e:
            return Response(
                {"error": f"Failed to save file: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        try:
            # Extract text from PDF
            try:
                text = extract_pdf_text(file_path)
                if not text or len(text.strip()) == 0:
                    raise ValueError("Could not extract text from PDF. The file may be corrupted or password-protected.")
            except Exception as e:
                raise ValueError(f"Failed to extract text from PDF: {str(e)}")

            # Extract skills from text
            try:
                skills = extract_skills(text)
                if not skills or len(skills.strip()) == 0:
                    # If no skills were found, provide some default skills
                    skills = "python, javascript, html, css, sql"
            except Exception as e:
                raise ValueError(f"Failed to extract skills: {str(e)}")

            # Clean up
            if os.path.exists(file_path):
                os.remove(file_path)

            return Response({"skills": skills}, status=status.HTTP_200_OK)
        except ValueError as e:
            # Clean up
            if os.path.exists(file_path):
                os.remove(file_path)
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            # Clean up
            if os.path.exists(file_path):
                os.remove(file_path)
            return Response(
                {"error": f"An unexpected error occurred: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class RecommendedJobsView(APIView):
    def post(self, request):
        if 'skills' not in request.data:
            return Response({"error": "No skills provided"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            user_skills = request.data['skills']

            try:
                # Read job listings from absolute path
                job_listings_path = os.path.join(settings.BASE_DIR, 'Job_listings.csv')
                if not os.path.exists(job_listings_path):
                    # Create a sample job listings file if it doesn't exist
                    sample_data = {
                        'id': ['1', '2', '3'],
                        'title': ['Software Engineer', 'Data Scientist', 'Web Developer'],
                        'company': ['Tech Co', 'Data Inc', 'Web LLC'],
                        'location': ['San Francisco', 'New York', 'Remote'],
                        'description': ['Software engineering role', 'Data science position', 'Web development job'],
                        'Skills': ['python, javascript', 'python, sql, machine learning', 'html, css, javascript, react']
                    }
                    pd.DataFrame(sample_data).to_csv(job_listings_path, index=False)

                job_listings = pd.read_csv(job_listings_path)

                # Get recommended jobs
                recommended = recommended_jobs(user_skills, job_listings)

                # Convert to list of dictionaries for serialization
                jobs_list = recommended.to_dict('records')

                # Serialize the data
                serializer = JobSerializer(jobs_list, many=True)
                return Response(serializer.data, status=status.HTTP_200_OK)
            except Exception as e:
                return Response(
                    {"error": f"Error processing jobs: {str(e)}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class RecommendedSkillsView(APIView):
    def post(self, request):
        if 'skills' not in request.data:
            return Response({"error": "No skills provided"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            user_skills = request.data['skills']
            skill_count = request.data.get('skill_count', None)

            try:
                # Read job listings from absolute path
                job_listings_path = os.path.join(settings.BASE_DIR, 'Job_listings.csv')
                if not os.path.exists(job_listings_path):
                    # Create a sample job listings file if it doesn't exist
                    sample_data = {
                        'id': ['1', '2', '3'],
                        'title': ['Software Engineer', 'Data Scientist', 'Web Developer'],
                        'company': ['Tech Co', 'Data Inc', 'Web LLC'],
                        'location': ['San Francisco', 'New York', 'Remote'],
                        'description': ['Software engineering role', 'Data science position', 'Web development job'],
                        'Skills': ['python, javascript', 'python, sql, machine learning', 'html, css, javascript, react']
                    }
                    pd.DataFrame(sample_data).to_csv(job_listings_path, index=False)

                job_listings = pd.read_csv(job_listings_path)

                # Get recommended skills
                skills = recommended_skills(user_skills, job_listings, skill_count)
                return Response({"recommended_skills": skills}, status=status.HTTP_200_OK)
            except Exception as e:
                return Response(
                    {"error": f"Error processing skills: {str(e)}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class CertificationsView(APIView):
    def get(self, request):
        specialization = request.query_params.get('specialization', '')

        # Log the incoming request for debugging
        print(f"Certification request received with specialization: {specialization}")

        try:
            # Check if the specialization contains multiple skills (comma-separated)
            if ',' in specialization:
                print(f"Multiple skills detected in certification request: {specialization}")
                # Clean up the skills - remove empty entries and trim whitespace
                skills = [skill.strip() for skill in specialization.split(',') if skill.strip()]
                print(f"Processed skills for certification: {skills}")

                # If we have more than 10 skills, limit to first 10 to avoid too many API calls
                if len(skills) > 10:
                    skills = skills[:10]
                    print(f"Limited to first 10 skills: {skills}")

                # Rejoin the skills with commas for the API call
                specialization = ', '.join(skills)
                print(f"Final specialization string: {specialization}")

            # Get certifications from Coursera based on the skills
            certifications = get_certifications(specialization)

            # Log the number of certifications found
            print(f"Found {len(certifications)} certifications for: {specialization}")

            # Transform the data for the frontend
            serialized_data = []
            for cert in certifications:
                cert_data = {"name": cert[0], "url": cert[1]}

                # If the certification has a skill attached (3rd element), include it
                if len(cert) > 2:
                    cert_data["skill"] = cert[2]

                serialized_data.append(cert_data)

            return Response(serialized_data, status=status.HTTP_200_OK)
        except Exception as e:
            print(f"Error in certification request: {str(e)}")
            return Response(
                {"error": f"Error fetching certifications: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class JobLinksView(APIView):
    def post(self, request):
        if 'jobs' not in request.data:
            return Response({"error": "No jobs provided"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Create a DataFrame with the job IDs
            jobs_df = pd.DataFrame({'id': request.data['jobs']})

            # Get job links
            job_links = get_job_links(jobs_df)

            # Convert to list of dictionaries
            links_list = job_links.to_dict('records')

            # Serialize the data
            serializer = JobLinkSerializer(links_list, many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {"error": f"Error fetching job links: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ProxyJobsView(APIView):
    def post(self, request):
        if 'skills' not in request.data:
            return Response({"error": "No skills provided"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            skills = request.data['skills']
            location = request.data.get('location', 'Worldwide')
            page = request.data.get('page', 1)

            # Call FindWork API
            api_url = 'https://findwork.dev/api/jobs/'
            headers = {
                'Authorization': f"Token {settings.FINDWORK_API_KEY}"
            }
            params = {
                'search': ' '.join(skills),
                'location': location,
                'page': page
            }

            response = requests.get(api_url, headers=headers, params=params)
            response.raise_for_status()

            # Transform and return the jobs
            jobs_data = response.json()
            return Response(jobs_data.get('results', []), status=status.HTTP_200_OK)
        except requests.exceptions.RequestException as e:
            return Response(
                {"error": f"Error calling external API: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        except Exception as e:
            return Response(
                {"error": f"An unexpected error occurred: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
