import argparse
import json
from pathlib import Path

from wave_sim.workflow_tools.compare_spectral_choppy_asset_bundles import summarize_bundle


def load_sweep_bundle_paths(sweep_manifest: Path) -> list[Path]:
    manifest = json.loads(sweep_manifest.read_text(encoding="utf-8"))
    return [Path(run["manifest"]) for run in manifest.get("runs", [])]


def normalize(value: float, values: list[float], invert: bool = False) -> float:
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        normalized = 1.0
    else:
        normalized = (value - minimum) / (maximum - minimum)
    return 1.0 - normalized if invert else normalized


def as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def evaluate_candidate(summary: dict, args: argparse.Namespace) -> dict:
    eta_range = as_float(summary["eta_range_max"])
    steepness = as_float(summary["steepness_p95_max"])
    foam_ratio = as_float(summary["foam_ratio_mean"])
    displacement = as_float(summary["horizontal_displacement_p95_max"])
    fold_ratio = as_float(summary["folded_triangle_ratio_max"])
    reject_reasons = []
    if fold_ratio > args.max_fold_ratio:
        reject_reasons.append(f"fold_ratio>{args.max_fold_ratio:g}")
    if displacement > args.max_displacement_p95:
        reject_reasons.append(f"disp_p95>{args.max_displacement_p95:g}")
    if foam_ratio < args.min_foam_ratio:
        reject_reasons.append(f"foam_ratio<{args.min_foam_ratio:g}")
    if foam_ratio > args.max_foam_ratio:
        reject_reasons.append(f"foam_ratio>{args.max_foam_ratio:g}")
    return {
        **summary,
        "eta_range": eta_range,
        "steepness": steepness,
        "foam_ratio": foam_ratio,
        "displacement": displacement,
        "fold_ratio": fold_ratio,
        "rejected": bool(reject_reasons),
        "reject_reasons": ", ".join(reject_reasons),
        "score": 0.0,
    }


def rank_candidates(candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []
    accepted = [candidate for candidate in candidates if not candidate["rejected"]]
    scoring_pool = accepted if accepted else candidates
    steepness_values = [candidate["steepness"] for candidate in scoring_pool]
    eta_values = [candidate["eta_range"] for candidate in scoring_pool]
    displacement_values = [candidate["displacement"] for candidate in scoring_pool]
    fold_values = [candidate["fold_ratio"] for candidate in scoring_pool]
    for candidate in scoring_pool:
        wave_score = 0.45 * normalize(candidate["steepness"], steepness_values) + 0.25 * normalize(candidate["eta_range"], eta_values)
        stability_score = 0.20 * normalize(candidate["displacement"], displacement_values, invert=True) + 0.10 * normalize(candidate["fold_ratio"], fold_values, invert=True)
        candidate["score"] = wave_score + stability_score
    return sorted(candidates, key=lambda candidate: (candidate["rejected"], -candidate["score"]))


def build_recommendation_report(candidates: list[dict]) -> str:
    best = next((candidate for candidate in candidates if not candidate["rejected"]), candidates[0] if candidates else None)
    lines = [
        "# Spectral Choppy Wave Asset Bundle Recommendation",
        "",
        "## Recommendation",
        "",
    ]
    if best is None:
        lines.append("- no candidates")
    else:
        lines.extend(
            [
                f"- selected_bundle: `{best['name']}`",
                f"- selected_manifest: `{best['manifest']}`",
                f"- score: `{best['score']:.6g}`",
                f"- rejected: `{best['rejected']}`",
                f"- reject_reasons: `{best['reject_reasons']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Candidates",
            "",
            "| Rank | Bundle | Rejected | Score | Choppiness | Foam Threshold | Eta Range Max | Steepness P95 Max | Foam Ratio Mean | Disp P95 Max | Fold Ratio Max | Reasons |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for index, candidate in enumerate(candidates, start=1):
        lines.append(
            "| {rank} | {name} | {rejected} | {score:.6g} | {choppiness} | {foam_threshold} | {eta:.6g} | {steepness:.6g} | {foam:.6g} | {disp:.6g} | {fold:.6g} | `{reasons}` |".format(
                rank=index,
                name=candidate["name"],
                rejected="yes" if candidate["rejected"] else "no",
                score=candidate["score"],
                choppiness=candidate["choppiness"],
                foam_threshold=candidate["foam_threshold"],
                eta=candidate["eta_range"],
                steepness=candidate["steepness"],
                foam=candidate["foam_ratio"],
                disp=candidate["displacement"],
                fold=candidate["fold_ratio"],
                reasons=candidate["reject_reasons"],
            )
        )
    return "\n".join(lines) + "\n"


def recommend_bundles(paths: list[Path], args: argparse.Namespace) -> dict:
    candidates = [evaluate_candidate(summarize_bundle(path), args) for path in paths]
    ranked = rank_candidates(candidates)
    best = next((candidate for candidate in ranked if not candidate["rejected"]), ranked[0] if ranked else None)
    return {
        "selected": best,
        "candidates": ranked,
        "thresholds": {
            "max_fold_ratio": args.max_fold_ratio,
            "max_displacement_p95": args.max_displacement_p95,
            "min_foam_ratio": args.min_foam_ratio,
            "max_foam_ratio": args.max_foam_ratio,
        },
    }


def write_recommendation(result: dict, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_recommendation_report(result["candidates"]), encoding="utf-8")
    json_output = output.with_suffix(".json")
    json_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recommend the best spectral choppy wave asset bundle from a sweep.")
    parser.add_argument("bundles", nargs="*", type=Path, help="Bundle directories or bundle_manifest.json paths.")
    parser.add_argument("--sweep-manifest", type=Path, default=None, help="Optional sweep_manifest.json to read bundle paths from.")
    parser.add_argument("--max-fold-ratio", type=float, default=0.0, help="Reject candidates above this folded triangle ratio.")
    parser.add_argument("--max-displacement-p95", type=float, default=0.35, help="Reject candidates above this p95 horizontal displacement.")
    parser.add_argument("--min-foam-ratio", type=float, default=0.0, help="Reject candidates below this mean foam ratio.")
    parser.add_argument("--max-foam-ratio", type=float, default=1.0, help="Reject candidates above this mean foam ratio.")
    parser.add_argument("--output", type=Path, default=Path("outputs/spectral_choppy_asset_bundle_recommendation.md"), help="Output Markdown path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = list(args.bundles)
    if args.sweep_manifest is not None:
        paths.extend(load_sweep_bundle_paths(args.sweep_manifest))
    if not paths:
        raise SystemExit("Provide at least one bundle path or --sweep-manifest.")
    result = recommend_bundles(paths, args)
    output = write_recommendation(result, args.output)
    print(f"Saved recommendation: {output}")
    if result["selected"] is not None:
        print(f"Selected: {result['selected']['name']}")


if __name__ == "__main__":
    main()
