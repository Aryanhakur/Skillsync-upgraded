SkillSync — Bridging the Gap Between Your Skills and Industry Expectations
SkillSync is an AI-powered job-matching platform. Upload your resume and it extracts your skills with a custom-trained NLP model, matches you against real job listings using TF-IDF + cosine similarity, shows you the skills you're missing for your target company, and recommends certifications to close the gap.

Built as a final-year capstone at Lovely Professional University (paper: "SkillSync — Bridging the gap between your skills and Industry Expectations").

Features
Resume skill extraction (PDF/DOCX) — a custom spaCy Named Entity Recognition model trained on annotated resume data, backed by a keyword matcher for robustness
Smart job matching — jobs ranked by TF-IDF + cosine similarity between your skills and each job's skill profile (live RapidAPI results or the bundled dataset — works with zero API keys)
Company skill trends — enter a target company (and optional role) and see the skills it demands most, with your gaps highlighted
Job match report — paste a job description or posting URL for an ATS-style coverage score with matched/missing skills
Skill-gap suggestions — the most in-demand market skills you don't have yet
Certification recommendations — relevant Coursera courses per skill, with a curated offline fallback
Accounts — JWT authentication, saved jobs, skills history, search history
CV template gallery — 20 ATS-friendly resume templates with thumbnails

How It Works
Resume (PDF/DOCX)
      │  pdfplumber / python-docx
      ▼
Plain text ──► spaCy NER model (output2/) ──► user skills
                                                │
Job listings (RapidAPI / Job_listings.csv)      │
      │  NER skill extraction per job           ▼
      └────────────► TF-IDF vectors ◄──── skills vector
                            │
                    cosine similarity
                            ▼
        ranked jobs · missing skills · company trends · certifications
        
  Tech Stack
_________________________________________________________________________________________________
| Layer        | Technology                                                                     |
| ---          | ---                                                                            |
| Backend      | Django 5 + Django REST Framework, SimpleJWT                                    |
| ML / NLP     | spaCy 3.8 (custom NER model), scikit-learn (TF-IDF, cosine similarity), pandas |
| Frontend     | Angular 19 (standalone components, static SPA build)                           |
| Data sources | RapidAPI jobs-api14, bundled Job_listings.csv, Coursera                        |
| Deployment   | Render (Blueprint or manual), or any static host for the frontend              |
|______________|________________________________________________________________________________|

Quick Start
1. Backend (port 8000)

cd pythonmodel
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # add your SECRET_KEY / API keys if any
python manage.py migrate
python manage.py runserver

The spaCy NER model requires spacy >=3.8.4,<3.9.0 (already pinned). Everything works without external API keys — job search automatically falls back to the bundled dataset.

2. Frontend (port 4200)

cd pythonmodel/job-portal-frontend
npm install
npm start                          # dev server at http://localhost:4200

Production build and serve:

npm run build                      # static output in dist/job-portal-frontend/browser
npm run serve:spa                  # serves the built SPA with history fallback

3. Tests

cd pythonmodel
python manage.py test job_matcher  # 23 engine + API tests

API Overview
______________________________________________________________________________________________
| Endpoint                 | Method | Purpose                                                 |
| ---                      | ---    | ---                                                     |
| /api/extract-skills/     | POST   | Skills from an uploaded resume (PDF/DOCX)               |
| /api/recommended-jobs/   | POST   | Dataset jobs ranked by similarity                       |
| /api/recommended-skills/ | POST   | In-demand skills you're missing                         |
| /api/proxy-jobs/         | POST   | Live job search (server-side key) with dataset fallback |
| /api/company-trends/     | POST   | Skill demand analysis for a target company              |
| /api/match-report/       | POST   | ATS-style match vs one job description/URL              |
| /api/certifications/     | GET    | Coursera course suggestions per skill                   |
| /auth/...                | —      | Register, login/refresh, profile, saved jobs, history   |
|__________________________|________|_________________________________________________________|

Full details: pythonmodel/README.md.

Configuration
All secrets live in backend environment variables (.env, gitignored — see .env.example). Key variables: SECRET_KEY, DEBUG, ALLOWED_HOSTS, CORS_ALLOWED_ORIGINS, RAPIDAPI_KEY (optional). The frontend holds no API keys; all external calls are proxied through Django.

Deployment
See DEPLOYMENT.md — covers the Render Blueprint and Blueprint-free options: manual Render services, or the built static frontend on Netlify/Vercel (routing files included) with the backend anywhere.

Security Notes
API keys are environment-only; never committed (rotate any key that ever touched git history)
Production fails fast on missing SECRET_KEY / ALLOWED_HOSTS; CORS is restricted outside debug mode
Job-URL fetching blocks private/internal hosts (SSRF guard)
License
MIT — see the LICENSE file.

Team
Vansh Mahajan · Animesh Kumar · Aryan Thakur · Mohit Kumar Gond · Gursimran Singh — School of Computer Applications, Lovely Professional University (guided by Ms. Ranjit Kaur Walia)
