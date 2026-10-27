# ATS Resume Checker

A Streamlit app that scores a resume for ATS (Applicant Tracking System) friendliness and suggests concrete improvements, powered by Google Gemini Flash.

## Features

- Upload a resume as **PDF, DOCX or TXT** (max 5 MB)
- Optionally paste a **job description** for a tailored keyword match
- **Overall ATS score (0-100)**, calculated from five weighted categories:
  keywords & relevance (30%), formatting & parseability (20%), content & impact (20%), structure & sections (15%), language & clarity (15%)
- Prioritised improvements (High / Medium / Low), strengths, missing keywords and before/after rewrite examples
- Download the full report as Markdown

## Run locally

1. Install Python 3.10 or newer.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey).
4. Provide the key, using either option:
   - Create `.streamlit/secrets.toml` containing:
     ```toml
     GEMINI_API_KEY = "your-key-here"
     ```
   - or just paste it into the sidebar when the app runs.
5. Start the app:
   ```bash
   streamlit run app.py
   ```

## Configuration

| Setting | Where | Default |
|---|---|---|
| `GEMINI_API_KEY` | Streamlit secrets or environment variable | none (sidebar input if missing) |
| `GEMINI_MODEL` | Streamlit secrets, environment variable, or sidebar | `gemini-3.5-flash` |

If Google retires or renames the model, change the model name in the sidebar or set `GEMINI_MODEL`.

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository (`app.py`, `requirements.txt`, `README.md` at the repo root).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app**, choose your repository and branch, and set the main file path to `app.py`.
4. Open **Advanced settings** and paste into **Secrets**:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
5. Click **Deploy**.

Never commit your API key to GitHub. The included `.gitignore` excludes `.streamlit/secrets.toml`.

## Notes and limitations

- The score is an AI-generated estimate, not the output of a real ATS. Use it as guidance; results can vary slightly between runs.
- Only the extracted text is analysed, so visual problems (columns, text boxes, images, icons) are inferred from how the text came out rather than seen directly.
- Scanned or image-only PDFs have no extractable text and cannot be analysed.
- Resume text is sent to the Gemini API for analysis. The app does not store it.
