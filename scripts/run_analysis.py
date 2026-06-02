from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algoradar import run_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AlgoRadar ML pipeline for a Codeforces handle.")
    parser.add_argument("handle", nargs="?", default="tourist")
    parser.add_argument("--refresh", action="store_true", help="Force Codeforces API refresh.")
    parser.add_argument("--embeddings", action="store_true", help="Try sentence-transformers/all-MiniLM-L6-v2.")
    parser.add_argument("--sample", action="store_true", help="Use reproducible local sample data instead of Codeforces API.")
    args = parser.parse_args()

    result = run_analysis(
        args.handle,
        force_refresh=args.refresh,
        prefer_transformer=args.embeddings,
        use_sample=args.sample,
    )

    print(f"Handle: {result.handle}")
    print(f"Source: {result.source}")
    print(f"Problems solved: {result.profile['problems_solved']}")
    print(f"Current rating: {result.profile['current_rating']}")
    print(f"Expected next contest band: {result.contest_model['predicted_band']}")
    print(f"Contest model: {result.contest_model['selected_model_name']}")
    print(f"Solve model: {result.solve_model['selected_model_name']}")
    print(f"Weakness model accuracy: {result.weakness_model['metrics']['accuracy']:.3f}")
    print(f"Recommendations: {len(result.recommendations)}")
    print()
    print("Top weakness tags:")
    print(result.weakness[["tag", "level", "accuracy", "recent_failures", "priority_score"]].head(8).to_string(index=False))
    print()
    print("Top recommendations:")
    print(result.recommendations[["problem_id", "name", "rating", "bucket", "solve_probability_pct"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
