# AlgoRadar

**AlgoRadar | AI Competitive Programming Weakness Analyzer and Problem Recommender**

AlgoRadar is a machine-learning powered competitive programming intelligence dashboard. A user enters a Codeforces handle, and the app analyzes real submission history to find weak topics, estimate solve probability, recommend problems, build a weekly practice roadmap, and track progress.

The goal is not just to show charts. AlgoRadar includes a real ML pipeline: API ingestion, data cleaning, feature engineering, baseline rules, train/test split, model metrics, problem ranking, and semantic similarity search.

## Features

- Live Codeforces handle analysis
- Rating-wise accuracy
- Tag-wise accuracy
- Most failed tags
- WA/TLE/runtime/compile verdict distribution
- Solved problem difficulty distribution
- Contest performance trend
- Weakness classifier for every tag
- Rule-based baseline compared with an ML classifier
- Next-contest performance band prediction
- Problem solve-probability model
- Confidence/growth/stretch recommendation queues
- Similar-but-harder problem retrieval
- Weekly practice roadmap
- Progress tracking
- Premium dark Streamlit UI

## Screens

1. User profile analytics
2. Weakness map
3. Problem recommendations
4. Solve probability
5. Weekly roadmap
6. Progress tracking

## Tech Stack

- Python
- Streamlit
- pandas
- scikit-learn
- Plotly
- Codeforces API
- TF-IDF vector search
- Optional Sentence Transformers embeddings

## ML Project Levels

| Level | Module | What it teaches |
| --- | --- | --- |
| Level 0 | ContestScore Predictor | train/test split, logistic regression, random forest, accuracy, precision, recall |
| Level 1 | CF Analytics Dashboard | API handling, data cleaning, visualization, analytics thinking |
| Level 2 | Weakness Classifier | feature design, rule baseline, ML comparison, explainability |
| Level 3 | Solve Probability Model | classification, probability buckets, feature importance |
| Level 4 | Problem Recommender | ranking, personalization, content-based recommendation |
| Level 5 | Similar Problem Finder | vector search, semantic similarity, nearest-neighbor retrieval |
| Level 6 | Final AlgoRadar | complete ML product with dashboard, recommendations, roadmap, and progress tracking |

## How It Works

AlgoRadar runs this pipeline:

1. Fetch Codeforces submissions, contest rating history, and problemset metadata.
2. Cache API responses locally in `data/cache/` so repeated runs are faster.
3. Convert raw API data into pandas feature tables.
4. Compute analytics such as rating accuracy, tag accuracy, verdict distribution, and contest trends.
5. Classify tag weakness using simple rules first.
6. Train a random forest weakness model and compare it with the rule baseline.
7. Train a next-contest performance band predictor.
8. Train a solve-probability model from user/problem features.
9. Rank unsolved problems into confidence, growth, and stretch buckets.
10. Build a similar-problem search layer using TF-IDF by default.

The app uses live Codeforces data by default. If the API is unavailable, the backend has a reproducible sample-data fallback so the pipeline can still be tested.

## Metric Definitions

AlgoRadar avoids treating every number as equally meaningful. The main user-facing metrics are:

- **Current rating**: official latest Codeforces rating.
- **Max rating**: official peak Codeforces rating.
- **Recent accuracy**: accepted-submission rate across the latest 80 submissions.
- **Rating-wise accuracy**: success rate grouped by official or estimated problem rating.
- **Hardest solved**: highest-rated accepted problem for a tag.
- **Repair priority**: a weakness score based on low accuracy, recent failures, and attempt volume.
- **Solve probability**: a monotonic scorecard estimate. AlgoRadar looks up the problem and automatically uses your recent failures on its tags. If the same user, tags, and solved count stay fixed, increasing the problem rating cannot increase the probability.
- **Rating source**: `official` means Codeforces provides the problem rating; `estimated` means AlgoRadar inferred it from problem index and solved count.

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/drawmebaaz/AlgoRadar.git
cd AlgoRadar
```

### 2. Create a Virtual Environment

#### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

#### Windows Command Prompt

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

#### Windows

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

#### macOS / Linux

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Run the App

#### Windows

```powershell
streamlit run app.py
```

#### macOS / Linux

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal. It is usually:

```text
http://localhost:8501
```

Enter a Codeforces handle and click **Analyze handle**.

## Run the ML Pipeline from Terminal

#### Windows

```powershell
python scripts\run_analysis.py tourist
```

#### macOS / Linux

```bash
python3 scripts/run_analysis.py tourist
```

Force a fresh Codeforces API pull:

#### Windows

```powershell
python scripts\run_analysis.py tourist --refresh
```

#### macOS / Linux

```bash
python3 scripts/run_analysis.py tourist --refresh
```

Run with local sample data:

#### Windows

```powershell
python scripts\run_analysis.py tourist --sample
```

#### macOS / Linux

```bash
python3 scripts/run_analysis.py tourist --sample
```

## Optional Embeddings

By default, AlgoRadar uses a fast TF-IDF vector search fallback for similar problems.

To use `sentence-transformers/all-MiniLM-L6-v2`, install the optional dependencies:

#### Windows

```powershell
python -m pip install -r requirements-embeddings.txt
```

#### macOS / Linux

```bash
python3 -m pip install -r requirements-embeddings.txt
```

Then run:

```bash
streamlit run app.py
```

Turn on **Use MiniLM embeddings** in the sidebar.

## Run Tests

Install dev dependencies:

#### Windows

```powershell
python -m pip install -r requirements-dev.txt
```

#### macOS / Linux

```bash
python3 -m pip install -r requirements-dev.txt
```

Run tests:

#### Windows

```powershell
python -m pytest -q
```

#### macOS / Linux

```bash
python3 -m pytest -q
```

## Free Deployment

### Streamlit Community Cloud

This is the easiest free permanent deployment.

1. Push this repository to GitHub.
2. Go to [https://share.streamlit.io](https://share.streamlit.io).
3. Sign in with GitHub.
4. Click **New app**.
5. Select this repository.
6. Set the main file path to:

```text
app.py
```

7. Deploy.

Streamlit Cloud will install `requirements.txt` and run the app.

### Temporary Public Link from Your Laptop

You can also expose your local Streamlit app with a tunnel such as Cloudflare Tunnel. This is free but temporary. The link works only while your computer is on and the tunnel process is running.

## Project Structure

```text
AlgoRadar/
  app.py                         Streamlit dashboard
  requirements.txt               App dependencies
  requirements-dev.txt           Test dependencies
  requirements-embeddings.txt    Optional MiniLM embedding dependencies
  runtime.txt                    Python runtime for cloud hosting
  algoradar/
    codeforces.py                Codeforces API client and caching
    config.py                    Project paths and constants
    features.py                  pandas feature engineering
    models.py                    contest and solve-probability models
    pipeline.py                  End-to-end analysis orchestration
    recommender.py               Problem ranking and recommendation logic
    sample_data.py               Reproducible fallback data
    semantic.py                  TF-IDF / MiniLM similar-problem retrieval
    weakness.py                  Rule baseline and ML weakness classifier
  scripts/
    run_analysis.py              CLI pipeline runner
  tests/
    test_pipeline.py             Smoke tests for the ML pipeline
  data/
    cache/                       Cached API responses
    models/                      Trained local model reports
```

## Solve Probability Buckets

AlgoRadar classifies recommended problems into:

- `>75%`: confidence problem
- `45-75%`: growth problem
- `25-45%`: stretch problem
- `<25%`: avoid for now

## Notes

- The dashboard uses real Codeforces data by default.
- API responses are cached locally to improve speed.
- The app analyzes the latest 2,500 submissions for a practical balance of speed and signal.
- The recommender prefilters the problemset before model scoring so the app stays responsive.
- The weakness classifier intentionally starts with rules before comparing with ML. This mirrors real ML workflows where a baseline matters.
