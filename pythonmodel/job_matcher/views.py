"""
Views for the job_matcher app.

API endpoints for extracting skills from resumes, recommending jobs and
skills, certification lookup, company trend analysis, match reports and
proxying live job searches.
"""
import logging
import os
import re
import tempfile
from pathlib import Path

import requests
from django.conf import settings
from rest_framework import status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from .serializers import JobSerializer, JobLinkSerializer

# Configure logging
logger = logging.getLogger(__name__)

try:
    from main import (
        extract_resume_text,
        extract_skills,
        recommended_jobs,
        recommended_skills,
        company_trend_analysis,
        job_match_report,
        get_certifications,
        get_job_links,
        load_job_listings,
        live_job_recommendations,
        fetch_job_description_from_url,
    )
    logger.info("Successfully imported functions from main.py")
except ImportError as e:  # pragma: no cover
    logger.error(f"Error importing functions from main.py: {e}")
    raise

ALLOWED_EXTENSIONS = (".pdf", ".docx")


def _get_listings():
    """Load the (cached, skills-precomputed) job listings."""
    return load_job_listings()


class ExtractSkillsView(APIView):
    """Extract skills from an uploaded resume (PDF or DOCX)."""
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        if "resume" not in request.FILES:
            return Response({"error": "No resume file provided"}, status=status.HTTP_400_BAD_REQUEST)

        resume_file = request.FILES["resume"]
        ext = Path(resume_file.name.lower()).suffix
        if ext not in ALLOWED_EXTENSIONS:
            return Response(
                {"error": "Only PDF and DOCX files are supported"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        tmp_path = None
        try:
            suffix = ext
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                for chunk in resume_file.chunks():
                    tmp.write(chunk)
                tmp_path = tmp.name

            text = extract_resume_text(tmp_path)
            if not text or not text.strip():
                raise ValueError(
                    "Could not extract text from the resume. "
                    "The file may be corrupted, password-protected, or a scanned image."
                )

            skills = extract_skills(text)
            logger.info(f"Extracted {len(skills.split(','))} skills from {resume_file.name}")
            return Response({"skills": skills}, status=status.HTTP_200_OK)

        except ValueError as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            logger.error(f"Skill extraction failed: {e}", exc_info=True)
            return Response(
                {"error": f"An unexpected error occurred: {e}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass


class RecommendedJobsView(APIView):
    """Recommend jobs from the local dataset, ranked by TF-IDF similarity."""

    def post(self, request):
        skills = request.data.get("skills")
        if not skills:
            return Response({"error": "No skills provided"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            limit = int(request.data.get("limit", 20))
        except (ValueError, TypeError):
            limit = 20

        try:
            listings = _get_listings()
            if listings.empty:
                return Response(
                    {"error": "Job listings dataset is empty"},
                    status=status.HTTP_503_SERVICE_UNAVAILABLE,
                )
            recommended = recommended_jobs(skills, listings, limit=limit)
            serializer = JobSerializer(recommended.to_dict("records"), many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error processing job recommendations: {e}", exc_info=True)
            return Response(
                {"error": f"Error processing jobs: {e}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class RecommendedSkillsView(APIView):
    """Suggest in-demand skills the user is missing."""

    def post(self, request):
        skills = request.data.get("skills")
        if not skills:
            return Response({"error": "No skills provided"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            skill_count = request.data.get("skill_count")
            if skill_count is not None:
                skill_count = int(skill_count)
        except (ValueError, TypeError):
            skill_count = None

        try:
            listings = _get_listings()
            if listings.empty:
                return Response(
                    {"error": "Job listings dataset is empty"},
                    status=status.HTTP_503_SERVICE_UNAVAILABLE,
                )
            result = recommended_skills(skills, listings, skill_count)
            return Response({"recommended_skills": result}, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error processing skill recommendations: {e}", exc_info=True)
            return Response(
                {"error": f"Error processing skills: {e}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class CompanyTrendsView(APIView):
    """
    Company-specific skill trend analysis: the skills a company demands most
    across its listings, and which of them the user is missing.
    """

    def post(self, request):
        company = request.data.get("company", "").strip()
        if not company:
            return Response({"error": "No company provided"}, status=status.HTTP_400_BAD_REQUEST)

        role = request.data.get("role", "").strip() or None
        skills = request.data.get("skills", "")

        try:
            listings = _get_listings()
            analysis = company_trend_analysis(skills, listings, company, role=role)
            if analysis.pop("found", True) is False:
                return Response(analysis, status=status.HTTP_404_NOT_FOUND)
            return Response(analysis, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error in company trend analysis: {e}", exc_info=True)
            return Response(
                {"error": f"Error analyzing company trends: {e}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class MatchReportView(APIView):
    """
    Compare the user's skills against a single job. Accepts either the job's
    skills/description directly, or a job posting URL to fetch and parse.
    """

    def post(self, request):
        skills = request.data.get("skills")
        if not skills:
            return Response({"error": "No skills provided"}, status=status.HTTP_400_BAD_REQUEST)

        job_skills = request.data.get("job_skills", "")
        job_url = request.data.get("job_url", "")

        try:
            if not job_skills and job_url:
                text = fetch_job_description_from_url(job_url)
                job_skills = extract_skills(text)
            elif not job_skills:
                job_description = request.data.get("job_description", "")
                if not job_description:
                    return Response(
                        {"error": "Provide job_skills, job_description or job_url"},
                        status=status.HTTP_400_BAD_REQUEST,
                    )
                job_skills = extract_skills(job_description)

            report = job_match_report(skills, job_skills)
            report["job_skills"] = [s.strip() for s in job_skills.split(",") if s.strip()]
            return Response(report, status=status.HTTP_200_OK)
        except ValueError as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except requests.exceptions.RequestException as e:
            return Response(
                {"error": f"Could not fetch the job posting: {e}"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        except Exception as e:
            logger.error(f"Error building match report: {e}", exc_info=True)
            return Response(
                {"error": f"Error building match report: {e}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class CertificationsView(APIView):
    """Certification recommendations (Coursera) for a skill or skill list."""

    def get(self, request):
        specialization = request.query_params.get("specialization", "")
        if not specialization:
            return Response(
                {"error": "No specialization provided"}, status=status.HTTP_400_BAD_REQUEST
            )

        try:
            certifications = get_certifications(specialization)
            serialized = []
            for cert in certifications:
                if not isinstance(cert, (list, tuple)) or len(cert) < 2:
                    continue
                entry = {"name": cert[0], "url": cert[1]}
                if len(cert) > 2:
                    entry["skill"] = cert[2]
                serialized.append(entry)
            return Response(serialized, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error fetching certifications: {e}", exc_info=True)
            return Response(
                {"error": f"Error fetching certifications: {e}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class JobLinksView(APIView):
    """Extract apply links from job provider data."""

    def post(self, request):
        jobs = request.data.get("jobs")
        if not isinstance(jobs, list):
            return Response(
                {"error": "Jobs must be a list of IDs"}, status=status.HTTP_400_BAD_REQUEST
            )

        try:
            listings = _get_listings()
            selected = listings[listings["id"].astype(str).isin([str(j) for j in jobs])]
            job_links = get_job_links(selected)
            serializer = JobLinkSerializer(job_links.to_dict("records"), many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error fetching job links: {e}", exc_info=True)
            return Response(
                {"error": f"Error fetching job links: {e}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class ProxyJobsView(APIView):
    """
    Job search endpoint used by the frontend. Tries live jobs from RapidAPI
    (key held server-side); falls back to TF-IDF matching over the local
    dataset so the portal works even without an API key. Responses always
    use the Job record shape the frontend expects.
    """

    def post(self, request):
        skills = request.data.get("skills")
        if not skills:
            return Response({"error": "No skills provided"}, status=status.HTTP_400_BAD_REQUEST)

        if isinstance(skills, str):
            skills = [s.strip() for s in skills.split(",") if s.strip()]
        if not isinstance(skills, list) or not skills:
            return Response(
                {"error": "Skills must be a list or comma-separated string"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        location = request.data.get("location", "Worldwide")
        remote_only = bool(request.data.get("remoteOnly", False))
        try:
            page = max(1, int(request.data.get("page", 1)))
        except (ValueError, TypeError):
            page = 1

        user_skills = request.data.get("user_skills", ", ".join(skills))

        try:
            # Live results (empty when no API key configured or API unreachable)
            jobs = live_job_recommendations(
                user_skills, skills, location=location, page=page, remote_only=remote_only
            )
            source = "live"

            if not jobs:
                listings = _get_listings()
                if listings.empty:
                    return Response(
                        {"error": "No job data available (dataset empty and live API unavailable)"},
                        status=status.HTTP_503_SERVICE_UNAVAILABLE,
                    )
                query = "|".join(re.escape(s) for s in skills)
                query_mask = listings["title"].fillna("").str.contains(query, case=False, regex=True)
                pool = listings[query_mask] if query_mask.any() else listings
                local = recommended_jobs(user_skills, pool, limit=20)
                jobs = JobSerializer(local.to_dict("records"), many=True).data
                source = "dataset"

            return Response({"source": source, "jobs": jobs}, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error in job search: {e}", exc_info=True)
            return Response(
                {"error": f"An unexpected error occurred: {e}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
