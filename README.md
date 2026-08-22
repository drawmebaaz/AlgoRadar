# AlgoRadar

![AlgoRadar combined analysis](social_assets/algoradar-screenshot-combined-analysis.png)

**AlgoRadar | Competitive Programming Analytics and Recommendation System**

AlgoRadar is a competitive programming analytics and recommendation system for Codeforces, CodeChef, and LeetCode. A user enters any combination of the three handles, and the app returns platform-wise analytics, focus areas, practice recommendations, and a solve-probability estimate for any problem.

Codeforces has the deepest signal because its official API exposes verdict-level submissions. LeetCode and CodeChef are normalized into the same product experience using their public profile, contest, topic, and practice-problem data.

## Features

- Optional Codeforces, CodeChef, and LeetCode handle inputs
- Live Codeforces handle analysis
- LeetCode profile, difficulty, topic, contest, and recommendation analysis
- CodeChef profile, rating trend, focus areas, and rating-band recommendations
- Combined cross-platform summary when at least two handles are provided
- Combined focus map
- Platform-separated practice queues
- Success rate by difficulty
- Tag-wise success and solved-depth analysis
- Most failed tags
- WA/TLE/runtime/compile verdict distribution
- Solved problem difficulty distribution
- Contest performance trend
- Focus table for every tag
- Transparent rule-based tag scoring
- Explainable solve-probability scorecard
- Confidence/growth/stretch recommendation queues
- Progress tracking
- Clean Streamlit dashboard

## Screens

1. Combined analysis
2. Codeforces
3. CodeChef
4. LeetCode
5. Recommendations
6. Solve estimate

## Visual Preview

### Combined Analysis

![Combined analysis dashboard](social_assets/algoradar-screenshot-combined-analysis.png)

### Recommendations Overview

![Recommendations overview](social_assets/algoradar-screenshot-recommendations-overview.png)

### Recommendation Table

![Recommendation table](social_assets/algoradar-screenshot-recommendations-table.png)

### Solve Estimate

![Solve estimate screen](social_assets/algoradar-screenshot-solve-probability.png)

## Tech Stack

- Python
- Streamlit
- pandas / NumPy
- scikit-learn
- Plotly
- Codeforces API
- LeetCode GraphQL public profile data
- CodeChef public profile and practice-problem data

## Solve-Probability Scorecard

AlgoRadar does not recommend problems by rating alone. Every recommendation is scored from a user-specific feature row built from the handle data available for that platform.

The solve estimate is a hand-tuned, explainable logistic scorecard built from:

- rating gap between the problem and the user's estimated level
- tag depth: how many problems on the relevant tags the user has already solved
- solved volume: total problems solved
- recent failures on the relevant tags
- problem popularity (accepted-submission count)
- calibrated difficulty: CodeChef and LeetCode ratings/labels are mapped onto a shared Codeforces-equivalent scale via `data/platform_calibration.csv`

There is no trained classifier behind this - the coefficients are fixed and chosen to keep the curve monotonic, so a harder problem never scores higher than an easier one, all else equal. Success rate is intentionally not the main signal, since public accepted-submission counts can be inflated by editorials or AI help; the scorecard weights solved volume, tag depth, and the gap between the target problem and the user's demonstrated level instead.

## Recommender

The recommender picks 60 unsolved problems per user, split into `confidence`, `growth`, and `stretch` queues:

1. **Candidate filtering** - the full problemset is narrowed to a rating window around the user's level and a tag-relevant pool (via Jaccard tag overlap against the user's weak tags) before scoring, so the app stays responsive.
2. **Solve-probability scoring** - each remaining candidate is scored with the scorecard above.
3. **Learning-value ranking** - candidates are ranked by a blend of solve-probability fit for the growth zone, tag relevance to the user's weak areas, evidence strength (how many tag-relevant problems the user has solved), problem popularity, and a Gaussian rating-closeness score.
4. **Similar-problem matching** - a tag vector is built from the user's solved problems, and each candidate's tag set is compared against it with scikit-learn's cosine similarity. This "topic familiarity" score nudges the ranking toward problems that build on topics the user already knows, without making the whole recommender just comfort-zone practice.
5. **Diversity** - the final picks are diversified across tags and rating bands so one topic or difficulty band doesn't dominate a queue.

The result is split into platform-specific queues:

- `confidence`: high-probability problems for momentum
- `growth`: sweet-spot problems for skill improvement
- `stretch`: harder problems that are still realistic

Codeforces uses the richest signal because it has official problem ratings, tags, solved counts, and verdict-level user submissions. CodeChef and LeetCode use the same scorecard against calibrated ratings from public profile, topic, contest, and difficulty signals.

## How It Works

AlgoRadar runs this pipeline:

1. Fetch Codeforces submissions, contest rating history, and problemset metadata (with a reproducible sample-data fallback if the API is unavailable).
2. Fetch optional LeetCode and CodeChef profiles only when their handles are added and their screens need the data.
3. Cache API/profile/problem-catalog responses locally in `data/cache/` so repeated runs are faster.
4. Convert raw platform data into normalized pandas feature tables.
5. Compute analytics such as success by difficulty, tag/topic coverage, verdict distribution, difficulty mix, and contest trends.
6. Score focus areas using transparent rules.
7. Score every candidate problem with the solve-probability scorecard.
8. Filter, rank, and diversify candidates into confidence/growth/stretch queues (see Recommender above).

## Metric Definitions

- **Current rating**: official latest rating for that platform only.
- **Max rating**: official peak rating for that platform only.
- **Platform rating**: a platform's own rating scale. AlgoRadar does not compare Codeforces, CodeChef, and LeetCode rating numbers as if they mean the same thing.
- **Shared difficulty scale**: an internal difficulty reference used only by solve estimate. CodeChef and LeetCode-style difficulties are mapped through `data/platform_calibration.csv`.
- **Recent success**: accepted-submission rate across the latest submissions.
- **Success by difficulty**: success rate grouped by official or estimated problem rating.
- **Hardest solved**: highest-rated accepted problem for a tag.
- **Focus score**: a topic score based on low success, recent failures, and attempt volume.
- **Solve estimate**: a practical estimate that prioritizes rating gap, solved volume, hardest solved difficulty, average solved difficulty, and the selected problem tags.
- **Topic familiarity**: a tag-vector cosine score that measures whether a recommended problem uses topics the user has already solved before. It is only a small comfort signal, not the whole recommender.
- **Rating fit**: a Gaussian rating-closeness score that is highest near the user's rating and smoothly falls for much easier or harder problems.
- **Rating source**: `official` means Codeforces provides the problem rating; `estimated` means AlgoRadar inferred it from problem index and solved count.
- **LeetCode problem reference**: LeetCode has Easy/Medium/Hard labels, topic tags, acceptance stats, and a frontend problem number. It does not expose an official problem rating or the original contest slot through the public problem lookup, so AlgoRadar lets the user add an optional Q1-Q4 contest-slot reference when known.
- **LeetCode focus areas**: coverage-based signal from public solved topic counters. LeetCode does not expose full public verdict history for every solved/failed problem.
- **CodeChef focus areas**: rating-history and practice-volume signal from the public profile. CodeChef public solved profiles do not expose reliable tag-level verdict history.
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
python -m pip install --upgrade -r requirements.txt
```

#### macOS / Linux

```bash
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade -r requirements.txt
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

## Run Analysis from Terminal

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

## Developer Notes / Linting

- Use `ruff` for lint checks and automatic fixes:

```bash
python -m pip install -r requirements-dev.txt
ruff check .
ruff --fix .
```

## Update README Screenshots

The README preview uses real app screenshots stored in `social_assets/`. To refresh the gallery, replace these files with new screenshots using the same filenames:

- `algoradar-screenshot-combined-analysis.png`
- `algoradar-screenshot-recommendations-overview.png`
- `algoradar-screenshot-recommendations-table.png`
- `algoradar-screenshot-solve-probability.png`

## Error Handling

If a Codeforces handle lookup fails - bad handle, network issue, rate limit, timeout - AlgoRadar shows the real reason as an error message. It never substitutes synthetic data for a failed live lookup, since that would silently misrepresent a real user's stats. Sample data is only ever used when explicitly requested, either via `--sample` on the CLI or directly through `algoradar.sample_data.make_sample_bundle` in tests/demos.

## Troubleshooting

Most setup errors come from running the app inside an existing Anaconda/base environment that already has old packages installed. The safest fix is to use the project virtual environment from the setup section.

If Streamlit shows an error like `unexpected keyword argument`, your environment is using an older preinstalled Streamlit package instead of the project dependencies. From the repository folder, upgrade the app dependencies:

#### Windows

```powershell
python -m pip install --upgrade -r requirements.txt
```

#### macOS / Linux

```bash
python3 -m pip install --upgrade -r requirements.txt
```

Then restart the app with `streamlit run app.py`.

For Anaconda users, avoid the base environment:

```bash
conda create -n algoradar python=3.12 -y
conda activate algoradar
python -m pip install --upgrade pip
python -m pip install --upgrade -r requirements.txt
streamlit run app.py
```

## Project Structure

```text
AlgoRadar/
  app.py                         Streamlit dashboard
  requirements.txt               App dependencies
  requirements-dev.txt           Test dependencies
  runtime.txt                    Python runtime version
  algoradar/
    codeforces.py                Codeforces API client and caching
    config.py                    Project paths and constants
    features.py                  pandas feature engineering
    models.py                    Explainable solve-probability scorecard
    pipeline.py                  End-to-end analysis orchestration
    platforms.py                 LeetCode/CodeChef clients, normalization, combined analysis
    recommender.py                Problem ranking and recommendation logic
    sample_data.py               Reproducible fallback data
    solve_probability.py         Cross-platform solve-estimate scorecard
    weakness.py                  Rule-based focus scoring
  scripts/
    run_analysis.py              CLI pipeline runner
  social_assets/
    algoradar-screenshot-combined-analysis.png
    algoradar-screenshot-recommendations-overview.png
    algoradar-screenshot-recommendations-table.png
    algoradar-screenshot-solve-probability.png
  tests/
    test_pipeline.py             Smoke tests for the analysis pipeline
    test_platforms.py            Non-network tests for platform normalization
    test_recommender.py          Recommendation scoring and ranking tests
    test_solve_probability.py    Cross-platform solve-estimate tests
  data/
    platform_calibration.csv     CodeChef/LeetCode difficulty calibration prior
    cache/                       Cached API responses
```

## Solve Estimate Buckets

AlgoRadar classifies recommended problems into:

- `>75%`: confidence problem
- `45-75%`: growth problem
- `25-45%`: stretch problem
- `<25%`: avoid for now

## Notes

- The dashboard uses real platform data by default. If a live lookup fails, AlgoRadar surfaces the actual error instead of substituting synthetic data.
- API responses, profile pages, and problem catalogs are cached locally to improve speed.
- Codeforces problem tags are pulled from the Codeforces problemset. LeetCode problem tags are pulled automatically when you enter a problem slug or URL in Solve estimate.
- First-time LeetCode/CodeChef recommendation pulls can take a few seconds because the app builds local caches. Reopening the same handles is much faster.
- The app analyzes the latest 2,500 Codeforces submissions for a practical balance of speed and signal.
- The recommender prefilters the problemset before scoring so the app stays responsive.
- Focus areas intentionally start with transparent rules so users can understand why a topic was marked important.
- README preview screenshots live in `social_assets/` and can be replaced with newer app screenshots when the UI changes.
