# Building AlgoRadar: Problems Faced and How We Solved Them

AlgoRadar started as a competitive-programming analytics idea and slowly became a full product-style ML project. The final version analyzes Codeforces, CodeChef, and LeetCode handles, builds platform-wise analytics, detects weak areas, recommends practice problems, and estimates solve probability.

This blog is a development log of the important problems we faced while building it, the mistakes that showed up, and the fixes that made the project more reliable.

## 1. The Project Was Becoming Only a UI

At the beginning, the risk was that AlgoRadar would become a polished dashboard without enough real machine-learning or data-engineering depth. A good UI is useful, but the main value of this project had to be the intelligence behind the recommendations.

The fix was to define the product around a real pipeline:

- fetch public platform data
- cache API responses
- clean and normalize submissions, problems, ratings, tags, and contest history
- create feature tables with pandas
- build weakness signals
- estimate solve probability
- rank personalized recommendations
- evaluate behavior with tests

This changed the project from "show charts" to "make decisions from user data."

## 2. Sample Data Was Not Enough

Early versions could run on sample data, but that was not enough for a real user. Users wanted to enter actual handles and see their own Codeforces, CodeChef, and LeetCode analysis.

The fix was to make live public data the default and keep sample data only as a fallback for testing. Codeforces became the strongest data source because its API exposes submissions, verdicts, ratings, problem tags, and problem ratings. CodeChef and LeetCode were added using the public data available from their profiles and problem catalogs.

The important design choice was to let missing handles be valid. If a user gives only Codeforces, the Codeforces section works. If they add two or three handles, combined analysis unlocks.

## 3. Platform Ratings Could Not Be Compared Directly

One early combined-analysis mistake was treating platform ratings as if they were on the same scale. A Codeforces rating, CodeChef rating, and LeetCode contest rating do not mean the same thing.

The fix was to remove cross-platform "best rating" style comparisons and keep native ratings platform-specific. Combined analysis now focuses on safer signals like total public solved count, available profiles, focus areas, and platform-specific weakness signals.

For solve probability, AlgoRadar uses a calibrated internal difficulty reference where needed, but the UI avoids telling users that the same numeric rating has the same meaning across all platforms.

## 4. "Average Rating" Was Not User-Friendly

A confusing metric appeared during review: a very strong user could have a solved-problem average rating much lower than their actual Codeforces rating. For example, a user rated above 3000 may still have an average solved rating around 1900 because they solved many easier problems over the years.

That metric was technically true but not very useful. It could make strong users think the product misunderstood them.

The fix was to prioritize current rating and max rating where platform ratings exist, and to use solved rating statistics only as internal model features. In the user-facing UI, AlgoRadar now tries to show metrics that a competitive programmer can immediately interpret.

## 5. Codeforces Problems Often Have Missing Ratings

Another issue appeared in solve probability and recommendations: not every Codeforces problem has an official rating. Asking the user to manually enter a problem rating was not good enough because Codeforces pages sometimes do not show one.

The fix was to estimate missing ratings using available metadata such as contest index and solved count. The app also tracks whether a rating is official or estimated. This rating-source confidence is used inside the solve-probability model so estimated ratings are treated more carefully than official ratings.

## 6. Solve Probability Was Overconfident

One serious ML/product issue was that a 1200-rated user could see a 1200-rated Codeforces problem with a probability close to 98 percent. That is not realistic. A same-rating problem should usually be a growth problem, not a guaranteed solve.

The model was also behaving strangely at one point: as problem rating increased, probability decreased at first but then increased later. That made no sense from a user perspective.

The fix was to rebuild the solve-probability scorecard with more CP-specific features and monotonic behavior. The model now uses 14 signals, including:

- problem rating or calibrated difficulty
- user rating
- rating gap
- tag accuracy
- attempts on related tags
- solved count on related tags
- average solved difficulty on related tags
- hardest solved difficulty on related tags
- total solved volume
- recent accuracy
- inferred recent failures
- problem popularity
- tag count
- rating-source confidence

Most importantly, probability is capped and calibrated using rating gap, solved depth, and tag ceiling. A harder problem should not randomly become easier just because of noisy features.

## 7. Accuracy Was Weighted Too Much

Accuracy sounds like an obvious signal, but in competitive programming it can be misleading. Many users solve after hints, editorials, AI assistance, or repeated attempts. That means accepted-submission accuracy does not always represent independent skill.

The fix was to downweight accuracy and give more importance to:

- how many problems the user solved
- what ratings they solved
- what tags they solved
- their hardest solved problem on the selected tags
- recent failures on similar tags

This made solve probability more useful for real practice planning.

## 8. The App Asked for Data It Already Had

The solve-probability screen originally asked users to manually enter recent failures on selected tags. That was bad UX because AlgoRadar already had submission history for Codeforces and could infer this.

The fix was to remove unnecessary manual inputs and infer recent failures from the user's data wherever possible. Users should only provide information the system cannot reasonably know.

## 9. Recommendations Needed More Than Rating Buckets

Early recommendations were too simple. A useful recommender should not only say "try rating 1400." It should understand weak tags, solved depth, problem popularity, estimated solve probability, and diversity.

The fix was to rank candidates using:

- solve-probability bucket fit
- weak-tag similarity
- hardest-solved ceiling gap
- evidence strength
- problem popularity
- rating confidence
- rating distance
- diversity across tags and rating bands

Recommendations are now separated by platform and bucket:

- confidence builders
- growth problems
- stretch problems

The recommender also avoids faking stretch problems. If the candidate pool does not contain true stretch-level problems, it is better to show fewer stretch items than to mislabel easy problems as stretch.

## 10. CodeChef and LeetCode Had Less Public Data

Codeforces provides rich public submission data. CodeChef and LeetCode expose less verdict-level public data, so it was not possible to copy the same analysis logic exactly.

The fix was to normalize each platform into the same product experience while respecting data limitations.

For LeetCode, AlgoRadar uses public profile stats, difficulty stats, topic counters, contest data, problem metadata, and recent accepted submissions where available. LeetCode does not have official numeric problem ratings like Codeforces, so difficulty is handled with Easy, Medium, Hard, and optional contest-position references.

For CodeChef, AlgoRadar uses public profile, rating trend, contests, and practice-problem data. The recommender uses dynamic scorecards instead of fixed hardcoded percentages.

## 11. Analyze Handles Felt Stuck

Some runs felt stuck after users entered handles and clicked Analyze. This was partly because the app was fetching multiple public sources, building recommendation catalogs, and sometimes loading optional embedding models in the same user action.

The fix was to reduce unnecessary work based on the active screen. For example, recommendations are only generated when the Recommendations screen needs them. API responses and problem catalogs are cached locally. The Codeforces submission limit was also kept practical so repeated analysis remains responsive.

## 12. The Problem Recommendation Page Failed for Some Users

One user reported that the recommendation page was not opening. This kind of issue usually comes from a bad assumption in the recommendation data frame: missing columns, empty recommendation sets, or a platform returning less data than expected.

The fix was to make the app handle empty and partial recommendation results more gracefully. Platform sections now show clear messages when handles are missing or when recommendations are not available yet.

## 13. Streamlit Version Differences Broke the App on Mac

A Mac user cloned the repo and hit this error:

```text
TypeError: ButtonMixin.button() got an unexpected keyword argument 'width'
```

The cause was a Streamlit API difference. Newer Streamlit versions support `width="stretch"`, but older versions expect `use_container_width=True`.

The fix was to add a compatibility helper that checks the function signature and passes the correct argument for the installed Streamlit version. This prevented the app from crashing across different environments.

## 14. NumPy 2.x Broke an Older Scientific Dependency

Another Mac/Anaconda user saw:

```text
AttributeError: np.unicode_ was removed in the NumPy 2.0 release
```

This was not an AlgoRadar logic bug. It came from an old dependency inside the user's Anaconda environment being mixed with NumPy 2.x.

The fix was to pin project dependencies and improve setup instructions. The README now recommends using a clean virtual environment instead of the Anaconda base environment. This avoids mixed package versions and makes installs reproducible.

## 15. MiniLM Embeddings Froze the App

When the MiniLM toggle was enabled, analysis appeared to freeze and PowerShell printed a huge amount of output. The root cause was that the app tried to import and load `sentence-transformers/all-MiniLM-L6-v2` during the Streamlit run.

On this machine, the embedding stack was also broken:

- `sentence-transformers` import hung
- `transformers` was partially uninstalled
- a bad `~ransformers` folder was left in `site-packages`
- orphaned Python processes kept running after timed-out attempts

The fix had several parts:

1. Stop the stuck Python processes.
2. Remove the broken `~ransformers` folders.
3. Pin a stable optional embedding stack:
   - `sentence-transformers==3.4.1`
   - `transformers==4.48.3`
   - `huggingface-hub==0.27.1`
   - `tokenizers==0.21.4`
4. Add `scripts/verify_minilm.py` so users can download and test MiniLM before using the Streamlit toggle.
5. Make the app use locally cached MiniLM by default and fall back to TF-IDF if MiniLM cannot load.

After the fix, MiniLM loaded correctly and produced `(2, 384)` embeddings with normalized vector norm `1.0`.

## 16. README Images Looked Too Synthetic

The first README images were generated promotional-style panels. They looked clean, but they did not show the real app. Later, actual screenshots were added for:

- combined analysis
- recommendations overview
- recommendation table
- solve probability

The fix was to replace the README preview with real screenshots stored in `social_assets/`. The old image generator script was removed because the README now depends on real app screenshots, not generated mockups.

## 17. Documentation Had to Match the Product

As the project evolved, the README needed several corrections. It originally mentioned old ideas like weekly roadmap and generic ML project levels. Those no longer matched the actual product.

The fix was to rewrite the README around the current app:

- supported platforms
- feature list
- real screenshots
- recommendation and probability logic
- setup commands for Windows, macOS, and Linux
- optional MiniLM setup
- troubleshooting common environment errors
- project structure

Good documentation became part of the product, not an afterthought.

## 18. Testing Had to Cover Product Behavior

The project needed tests that checked more than whether functions returned something. The important bugs were behavioral:

- probability should decrease as problem rating increases
- missing Codeforces ratings should be estimated
- feature rows should include solved-depth signals
- external platform helpers should return user-sensitive probabilities
- the Streamlit app should start without exceptions

The final test suite covered these behaviors, and the app was also smoke-tested with Streamlit's testing API.

## What This Project Taught

AlgoRadar became a strong project because it combined multiple hard parts:

- real API handling
- caching
- pandas feature engineering
- platform normalization
- model-style scoring
- recommendation ranking
- semantic retrieval
- Streamlit UI
- environment debugging
- README and setup polish

The most important lesson was that an ML product is not only about the model. The product becomes useful when the data is reliable, the metrics make sense to users, the UI avoids vague numbers, the system handles missing data, and the setup works on other people's machines.

## Final State

AlgoRadar now works as a multi-platform competitive-programming intelligence dashboard. It can analyze real handles, produce platform-specific insights, recommend problems by confidence/growth/stretch buckets, estimate solve probability, and optionally use MiniLM embeddings for similar-problem retrieval.

The biggest improvement was moving from a dashboard that displayed data to a system that explains what the user should practice next and why.
