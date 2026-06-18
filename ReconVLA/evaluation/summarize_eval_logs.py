#!/usr/bin/env python3
"""Summarize CALVIN rollout evaluation logs."""

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence


DEFAULT_EVAL_LOG_ROOT = Path(
    "/home/share/ltwwa4al/home/Huangjian_Hust/text_recon/eval_logs"
)


ANSI_ESCAPE_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
NUMBER = r"(?:\d+(?:\.\d+)?)"
PROGRESS_RE = re.compile(
    rf"1/5\s*:\s*({NUMBER})%\s*\|\s*"
    rf"2/5\s*:\s*({NUMBER})%\s*\|\s*"
    rf"3/5\s*:\s*({NUMBER})%\s*\|\s*"
    rf"4/5\s*:\s*({NUMBER})%\s*\|\s*"
    rf"5/5\s*:\s*({NUMBER})%[^\r\n]*?\b(\d+)\s*/\s*(\d+)\b"
)
AVERAGE_RE = re.compile(
    rf"^Average successful sequence length:\s*({NUMBER})\s*$", re.MULTILINE
)
FINAL_SR_RE = re.compile(rf"^\s*([1-5]):\s*({NUMBER})%\s*$", re.MULTILINE)
TASK_RE = re.compile(
    rf"^\s*([A-Za-z0-9_]+):\s*(\d+)\s*/\s*(\d+)\s*\|\s*SR:\s*({NUMBER})%\s*$",
    re.MULTILINE,
)


@dataclass
class TaskMetric:
    task: str
    successes: int
    total: int
    success_rate: float


@dataclass
class EvalResult:
    experiment: str
    log_path: Path
    status: str
    completed_sequences: Optional[int] = None
    total_sequences: Optional[int] = None
    average_successful_sequence_length: Optional[float] = None
    success_rates: Dict[int, float] = field(default_factory=dict)
    task_metrics: List[TaskMetric] = field(default_factory=list)


def _clean_log_text(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text).replace("\x00", "")


def parse_eval_log(text: str, experiment: str, log_path: Path) -> EvalResult:
    """Parse one evaluator stdout log into structured metrics."""
    clean_text = _clean_log_text(text)

    progress_matches = list(PROGRESS_RE.finditer(clean_text))
    progress_rates: Dict[int, float] = {}
    completed_sequences = None
    total_sequences = None
    if progress_matches:
        latest = progress_matches[-1]
        progress_rates = {
            index: float(latest.group(index)) for index in range(1, 6)
        }
        completed_sequences = int(latest.group(6))
        total_sequences = int(latest.group(7))

    average_match = AVERAGE_RE.search(clean_text)
    average = float(average_match.group(1)) if average_match else None

    final_rates = {
        int(match.group(1)): float(match.group(2))
        for match in FINAL_SR_RE.finditer(clean_text)
    }
    # The current evaluator appends results_avg to its 500 per-sequence values
    # before printing the final SR block, making that block use a denominator of
    # 501. A complete 500/500 tqdm snapshot is therefore the authoritative SR.
    progress_is_complete = (
        completed_sequences is not None
        and total_sequences is not None
        and completed_sequences == total_sequences
    )
    if progress_is_complete and len(progress_rates) == 5:
        success_rates = progress_rates
    elif len(final_rates) == 5:
        success_rates = final_rates
    else:
        success_rates = progress_rates

    task_metrics = [
        TaskMetric(
            task=match.group(1),
            successes=int(match.group(2)),
            total=int(match.group(3)),
            success_rate=float(match.group(4)),
        )
        for match in TASK_RE.finditer(clean_text)
    ]

    if average is not None and len(final_rates) == 5:
        status = "complete"
    elif success_rates:
        status = "incomplete"
    else:
        status = "unparseable"

    return EvalResult(
        experiment=experiment,
        log_path=log_path,
        status=status,
        completed_sequences=completed_sequences,
        total_sequences=total_sequences,
        average_successful_sequence_length=average,
        success_rates=success_rates,
        task_metrics=task_metrics,
    )


def collect_eval_results(root: Path) -> List[EvalResult]:
    """Recursively find and parse every file named eval.log below root."""
    results = []
    for log_path in sorted(root.rglob("eval.log"), key=lambda path: path.as_posix()):
        text = log_path.read_text(encoding="utf-8", errors="replace")
        results.append(
            parse_eval_log(
                text,
                experiment=log_path.parent.name,
                log_path=log_path,
            )
        )
    return sorted(results, key=lambda result: result.experiment)


def _summary_row(result: EvalResult) -> Dict[str, object]:
    row: Dict[str, object] = {
        "experiment": result.experiment,
        "status": result.status,
        "completed_sequences": result.completed_sequences,
        "total_sequences": result.total_sequences,
        "average_successful_sequence_length": result.average_successful_sequence_length,
        "eval_log": str(result.log_path),
    }
    for index in range(1, 6):
        row[f"sr_{index}"] = result.success_rates.get(index)
    return row


def _result_dict(result: EvalResult) -> Dict[str, object]:
    return {
        "experiment": result.experiment,
        "status": result.status,
        "completed_sequences": result.completed_sequences,
        "total_sequences": result.total_sequences,
        "average_successful_sequence_length": result.average_successful_sequence_length,
        "success_rates": {
            str(index): rate for index, rate in sorted(result.success_rates.items())
        },
        "task_metrics": [
            {
                "task": metric.task,
                "successes": metric.successes,
                "total": metric.total,
                "success_rate": metric.success_rate,
            }
            for metric in result.task_metrics
        ],
        "eval_log": str(result.log_path),
    }


def write_reports(
    results: Sequence[EvalResult], output_dir: Path, source_root: Path
) -> Dict[str, Path]:
    """Write experiment CSV, task CSV, and JSON reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary_csv": output_dir / "eval_summary.csv",
        "task_csv": output_dir / "eval_task_metrics.csv",
        "json": output_dir / "eval_summary.json",
    }

    summary_fields = [
        "experiment",
        "status",
        "completed_sequences",
        "total_sequences",
        "average_successful_sequence_length",
        "sr_1",
        "sr_2",
        "sr_3",
        "sr_4",
        "sr_5",
        "eval_log",
    ]
    with paths["summary_csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for result in results:
            writer.writerow(_summary_row(result))

    task_fields = ["experiment", "task", "successes", "total", "success_rate"]
    with paths["task_csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=task_fields)
        writer.writeheader()
        for result in results:
            for metric in result.task_metrics:
                writer.writerow(
                    {
                        "experiment": result.experiment,
                        "task": metric.task,
                        "successes": metric.successes,
                        "total": metric.total,
                        "success_rate": metric.success_rate,
                    }
                )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "runs": [_result_dict(result) for result in results],
    }
    paths["json"].write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return paths


def format_summary_table(results: Sequence[EvalResult]) -> str:
    """Render a compact human-readable model comparison table."""
    headers = ["experiment", "status", "progress", "avg_len", "SR@1", "SR@2", "SR@3", "SR@4", "SR@5"]
    rows = []
    for result in results:
        if result.completed_sequences is not None and result.total_sequences is not None:
            progress = f"{result.completed_sequences}/{result.total_sequences}"
        else:
            progress = "-"
        average = (
            f"{result.average_successful_sequence_length:.3f}"
            if result.average_successful_sequence_length is not None
            else "-"
        )
        rows.append(
            [
                result.experiment,
                result.status,
                progress,
                average,
                *[
                    f"{result.success_rates[index]:.1f}%"
                    if index in result.success_rates
                    else "-"
                    for index in range(1, 6)
                ],
            ]
        )

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def render(row: Sequence[str]) -> str:
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))

    separator = "  ".join("-" * width for width in widths)
    return "\n".join([render(headers), separator, *(render(row) for row in rows)])


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize CALVIN eval.log files into CSV and JSON reports."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_EVAL_LOG_ROOT,
        help=f"Eval-log root to scan (default: {DEFAULT_EVAL_LOG_ROOT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Report directory (default: the eval-log root)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    root = args.root
    if not root.is_dir():
        print(f"[ERROR] Eval-log root does not exist: {root}", file=sys.stderr)
        return 2

    results = collect_eval_results(root)
    if not results:
        print(f"[ERROR] No eval.log files found below: {root}", file=sys.stderr)
        return 1

    output_dir = args.output_dir if args.output_dir is not None else root
    paths = write_reports(results, output_dir=output_dir, source_root=root)
    print(format_summary_table(results))
    print("\nReports:")
    for path in paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
