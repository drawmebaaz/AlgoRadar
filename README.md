# AlgoRadar

**AlgoRadar | AI Competitive Programming Weakness Analyzer and Problem Recommender**

AlgoRadar is a machine-learning powered competitive programming intelligence dashboard for Codeforces, CodeChef, and LeetCode. A user enters any combination of the three handles, and the app returns platform-wise analytics, weakness signals, recommendations, and solve-probability estimates from the handles provided.

The goal is not just to show charts. AlgoRadar includes a real ML pipeline where the platform data supports it: API ingestion, data cleaning, feature engineering, baseline rules, train/test split, model metrics, problem ranking, and semantic similarity search. Codeforces has the deepest model because its official API exposes verdict-level submissions. LeetCode and CodeChef are normalized into the same product experience using their public profile, contest, topic, and practice-problem data.

## Features

- Optional Codeforces, CodeChef, and LeetCode handle inputs
- Live Codeforces handle analysis
- LeetCode profile, difficulty, topic, contest, and recommendation analysis
- CodeChef profile, rating trend, weakness signals, and rating-band recommendations
- Combined cross-platform summary when at least two handles are provided
- Combined weakness/focus map
- Platform-separated practice queues
- Rating-wise accuracy
- Tag-wise accuracy
- Most failed tags
- WA/TLE/runtime/compile verdict distribution
- Solved problem difficulty distribution
- Contest performance trend
- Weakness classifier for every tag
- Rule-based baseline compared with an ML classifier
- Next-contest performance band prediction
- Calibrated problem solve-probability model
- Confidence/growth/stretch recommendation queues
- Similar-but-harder problem retrieval
- Progress tracking
- Premium dark Streamlit UI

## Screens

1. Combined analysis
2. Codeforces
3. CodeChef
4. LeetCode
5. Recommendations
6. Solve probability

## Tech Stack

- Python
- Streamlit
- pandas
- scikit-learn
- Plotly
- Codeforces API
- LeetCode GraphQL public profile data
- CodeChef public profile and practice-problem data
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
| Level 6 | Final AlgoRadar | complete ML product with dashboard, recommendations, solve probability, and progress tracking |

## How It Works

AlgoRadar runs this pipeline:

1. Fetch Codeforces submissions, contest rating history, and problemset metadata.
2. Fetch optional LeetCode and CodeChef profiles only when their handles are added and their screens need the data.
3. Cache API/profile/problem-catalog responses locally in `data/cache/` so repeated runs are faster.
4. Convert raw platform data into normalized pandas feature tables.
5. Compute analytics such as rating accuracy, tag/topic coverage, verdict distribution, difficulty mix, and contest trends.
6. Classify weakness using simple rules first.
7. Train a random forest weakness model and compare it with the rule baseline for Codeforces.
8. Train a next-contest performance band predictor.
9. Estimate solve probability from solved volume and solved-difficulty strength on relevant tags, using all provided handles where possible.
10. Rank problems into confidence, growth, and stretch buckets.
11. Build a similar-problem search layer using TF-IDF by default.

The app uses live Codeforces data by default. If the API is unavailable, the backend has a reproducible sample-data fallback so the pipeline can still be tested.

## Metric Definitions

AlgoRadar avoids treating every number as equally meaningful. The main user-facing metrics are:

- **Current rating**: official latest rating for that platform only.
- **Max rating**: official peak rating for that platform only.
- **Native rating**: a platform's own rating scale. AlgoRadar does not compare Codeforces, CodeChef, and LeetCode native ratings as equal numbers.
- **CF-equivalent difficulty**: an internal calibrated difficulty reference used only by solve probability. CodeChef and LeetCode-style numeric difficulties are mapped through `data/platform_calibration.csv`.
- **Recent accuracy**: accepted-submission rate across the latest 80 submissions.
- **Rating-wise accuracy**: success rate grouped by official or estimated problem rating.
- **Hardest solved**: highest-rated accepted problem for a tag.
- **Repair priority**: a weakness score based on low accuracy, recent failures, and attempt volume.
- **Solve probability**: a calibrated scorecard estimate that prioritizes solved volume, hardest solved difficulty, average solved difficulty, and the selected problem tags. Accuracy is intentionally not a primary signal because accepted submissions can be distorted by AI/editorial help.
- **Rating source**: `official` means Codeforces provides the problem rating; `estimated` means AlgoRadar inferred it from problem index and solved count.
- **LeetCode problem reference**: LeetCode has Easy/Medium/Hard labels, topic tags, acceptance stats, and a frontend problem number. It does not expose an official problem rating or the original contest slot through the public problem lookup, so AlgoRadar lets the user add an optional Q1-Q4 contest-slot reference when known.
- **LeetCode topic weakness**: coverage-based signal from public solved topic counters. LeetCode does not expose full public verdict history for every solved/failed problem.
- **CodeChef weakness**: rating-history and practice-volume signal from the public profile. CodeChef public solved profiles do not expose reliable tag-level verdict history.
- **Combined solved**: sum of public solved-count signals across connected platforms. It is useful for progress direction, not as a perfect apples-to-apples skill rating.

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

Enter one or more platform handles and click **Analyze handles**. Missing handles are allowed; the related platform section will ask you to add that handle first.

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
    platforms.py                 LeetCode/CodeChef clients, normalization, combined analysis
    recommender.py               Problem ranking and recommendation logic
    sample_data.py               Reproducible fallback data
    semantic.py                  TF-IDF / MiniLM similar-problem retrieval
    solve_probability.py         Cross-platform solve-probability scorecard
    weakness.py                  Rule baseline and ML weakness classifier
  scripts/
    run_analysis.py              CLI pipeline runner
  tests/
    test_pipeline.py             Smoke tests for the ML pipeline
    test_platforms.py            Non-network tests for platform normalization
    test_solve_probability.py    Cross-platform probability scorecard tests
  data/
    platform_calibration.csv     CodeChef/LeetCode difficulty to CF-equivalent calibration prior
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

- The dashboard uses real platform data by default.
- API responses, profile pages, and problem catalogs are cached locally to improve speed.
- Solve probability uses `data/platform_calibration.csv` as a calibration prior, not as ground-truth solved labels. It should be retrained later with real common-user solve outcomes if you collect that dataset.
- Codeforces problem tags are pulled from the Codeforces problemset. LeetCode problem tags are pulled automatically when you enter a problem slug or URL in Solve probability.
- First-time LeetCode/CodeChef recommendation pulls can take a few seconds because the app builds local caches. Reopening the same handles is much faster.
- The app analyzes the latest 2,500 Codeforces submissions for a practical balance of speed and signal.
- The recommender prefilters the problemset before model scoring so the app stays responsive.
- The weakness classifier intentionally starts with rules before comparing with ML. This mirrors real ML workflows where a baseline matters.
