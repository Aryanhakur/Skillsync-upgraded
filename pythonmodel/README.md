# SkillSync - Job Matching Platform

A platform that extracts skills from resumes and matches them with real-time job listings.

## Features

- Resume skill extraction (PDF **and DOCX**) via a custom spaCy NER model (`output2/`)
- Job matching with TF-IDF + cosine similarity (local dataset + live RapidAPI jobs)
- Recommended skills based on job market demand
- Certification recommendations (Coursera scraping with a curated offline fallback)
- **Company skill trend analysis** — see which skills a target company demands most and which you're missing
- **Job match report** — paste a job description (or posting URL) for an ATS-style coverage score
- User authentication (JWT) with saved jobs, skills history and search history

## API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/extract-skills/` | POST (multipart `resume`) | Extract skills from an uploaded PDF/DOCX |
| `/api/recommended-jobs/` | POST `{skills, limit?}` | Rank local dataset jobs by similarity |
| `/api/recommended-skills/` | POST `{skills, skill_count?}` | In-demand skills the user is missing |
| `/api/proxy-jobs/` | POST `{skills, location?, page?, remoteOnly?}` | Live RapidAPI jobs (server-side key) with dataset fallback |
| `/api/company-trends/` | POST `{company, role?, skills?}` | Company-specific skill demand analysis |
| `/api/match-report/` | POST `{skills, job_description? \| job_url?}` | ATS-style match report for one job |
| `/api/certifications/` | GET `?specialization=` | Coursera certification suggestions |
| `/api/job-links/` | POST `{jobs: [ids]}` | Apply links for job IDs |
| `/auth/...` | various | Register, login/refresh, profile, saved jobs, history |

## Setup

### Backend

```bash
cd pythonmodel
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then edit .env (see below)
python manage.py migrate
python manage.py runserver       # http://localhost:8000
```

Note: the spaCy NER model requires **spacy >=3.8.4,<3.9.0** (pinned in requirements.txt).

### Frontend

```bash
cd pythonmodel/job-portal-frontend
npm install
npm start                        # dev server on http://localhost:4200
```

Production build + static serving:

```bash
npm run build                    # static SPA into dist/
npm run serve:spa                # serves the SPA with history fallback
```

The frontend is a static SPA (no SSR). All external APIs are called through the
Django backend, so no API keys exist in frontend code.

### Tests

```bash
cd pythonmodel
python manage.py test job_matcher   # 23 engine + API tests
```

## Environment Variables (backend `.env`)

- `SECRET_KEY` — Django secret key (generate one; required when `DEBUG=False`)
- `DEBUG` — `True` for development, `False` for production
- `RAPIDAPI_KEY` — your RapidAPI key for jobs-api14 (optional: without it the app
  falls back to TF-IDF matching over `Job_listings.csv`)
- `RAPIDAPI_HOST` — default `jobs-api14.p.rapidapi.com`
- `ALLOWED_HOSTS` — comma-separated, required when `DEBUG=False`
- `CORS_ALLOWED_ORIGINS` — comma-separated frontend origins, required when `DEBUG=False`

## Data

`Job_listings.csv` holds the local job dataset. A `Skills` column is
auto-computed with the NER model on first load and persisted back to the CSV;
subsequent loads are fast. Delete the column (or the file) to recompute.

## Security Notes

- API keys live only in backend environment variables — never in code or git
- `.env` is gitignored; use `.env.example` as the template
- Job-URL fetching in the match report blocks private/internal hosts (SSRF guard)
