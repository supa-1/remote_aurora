import csv
import json
import tempfile
import unittest
from pathlib import Path

from ReconVLA.evaluation.summarize_eval_logs import (
    collect_eval_results,
    parse_eval_log,
    write_reports,
)


class ParseEvalLogTests(unittest.TestCase):
    def test_parses_completed_eval_summary_and_task_metrics(self):
        text = """Gym has been unmaintained since 2022.
logging to /tmp/eval_logs/ckpt10388_task_D_D
1/5 : 93.4% | 2/5 : 83.8% | 3/5 : 74.2% | 4/5 : 65.0% | 5/5 : 59.0% ||: 100%|x| 500/500 [33:30:52<00:00]
results_avg: 3.754
Results for Epoch None:
Average successful sequence length: 3.754
Success rates for i instructions in a row:
1: 93.2%
2: 83.6%
3: 74.1%
4: 64.9%
5: 58.9%
rotate_blue_block_right: 32 / 34 |  SR: 94.1%
move_slider_right: 129 / 130 |  SR: 99.2%
Best model: epoch None with average sequences length of 3.754
"""

        result = parse_eval_log(
            text,
            experiment="ckpt10388_task_D_D",
            log_path=Path("/tmp/eval_logs/ckpt10388_task_D_D/eval.log"),
        )

        self.assertEqual(result.status, "complete")
        self.assertEqual(result.completed_sequences, 500)
        self.assertEqual(result.total_sequences, 500)
        self.assertAlmostEqual(result.average_successful_sequence_length, 3.754)
        self.assertEqual(
            result.success_rates,
            {1: 93.4, 2: 83.8, 3: 74.2, 4: 65.0, 5: 59.0},
        )
        self.assertEqual(len(result.task_metrics), 2)
        self.assertEqual(result.task_metrics[0].task, "rotate_blue_block_right")
        self.assertEqual(result.task_metrics[0].successes, 32)
        self.assertEqual(result.task_metrics[0].total, 34)
        self.assertAlmostEqual(result.task_metrics[0].success_rate, 94.1)

    def test_uses_last_progress_snapshot_for_incomplete_eval(self):
        text = """startup noise
1/5 : 90.0% | 2/5 : 80.0% | 3/5 : 70.0% | 4/5 : 60.0% | 5/5 : 50.0% ||:  20%|x| 100/500 [01:00<04:00]
1/5 : 91.1% | 2/5 : 81.2% | 3/5 : 71.3% | 4/5 : 61.4% | 5/5 : 51.5% ||:  25%|x| 125/500 [01:15<03:45]
"""

        result = parse_eval_log(
            text,
            experiment="task_D_D_lora_plain_3000",
            log_path=Path("/tmp/task_D_D_lora_plain_3000/eval.log"),
        )

        self.assertEqual(result.status, "incomplete")
        self.assertEqual(result.completed_sequences, 125)
        self.assertEqual(result.total_sequences, 500)
        self.assertIsNone(result.average_successful_sequence_length)
        self.assertEqual(
            result.success_rates,
            {1: 91.1, 2: 81.2, 3: 71.3, 4: 61.4, 5: 51.5},
        )
        self.assertEqual(result.task_metrics, [])

    def test_marks_log_without_metrics_as_unparseable(self):
        result = parse_eval_log(
            "Gym warning only\nConnection Error.\n",
            experiment="broken_run",
            log_path=Path("/tmp/broken_run/eval.log"),
        )

        self.assertEqual(result.status, "unparseable")
        self.assertEqual(result.success_rates, {})
        self.assertEqual(result.task_metrics, [])


class ReportTests(unittest.TestCase):
    COMPLETED_LOG = """1/5 : 90.0% | 2/5 : 80.0% | 3/5 : 70.0% | 4/5 : 60.0% | 5/5 : 50.0% ||: 100%|x| 500/500 [01:00<00:00]
Average successful sequence length: 3.5
Success rates for i instructions in a row:
1: 90.0%
2: 80.0%
3: 70.0%
4: 60.0%
5: 50.0%
open_drawer: 9 / 10 | SR: 90.0%
"""

    def test_collects_only_eval_logs_and_uses_parent_directory_as_experiment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_b = root / "run_b"
            run_a = root / "nested" / "run_a"
            run_b.mkdir()
            run_a.mkdir(parents=True)
            (run_b / "eval.log").write_text(self.COMPLETED_LOG, encoding="utf-8")
            (run_b / "server.log").write_text("ignored", encoding="utf-8")
            (run_a / "eval.log").write_text("startup only", encoding="utf-8")

            results = collect_eval_results(root)

        self.assertEqual([result.experiment for result in results], ["run_a", "run_b"])
        self.assertEqual([result.status for result in results], ["unparseable", "complete"])

    def test_writes_experiment_task_and_json_reports(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "run_a"
            output_dir = root / "reports"
            run_dir.mkdir()
            log_path = run_dir / "eval.log"
            log_path.write_text(self.COMPLETED_LOG, encoding="utf-8")
            results = collect_eval_results(root)

            paths = write_reports(results, output_dir=output_dir, source_root=root)

            with paths["summary_csv"].open(newline="", encoding="utf-8") as handle:
                summary_rows = list(csv.DictReader(handle))
            with paths["task_csv"].open(newline="", encoding="utf-8") as handle:
                task_rows = list(csv.DictReader(handle))
            report = json.loads(paths["json"].read_text(encoding="utf-8"))

        self.assertEqual(summary_rows[0]["experiment"], "run_a")
        self.assertEqual(summary_rows[0]["status"], "complete")
        self.assertEqual(summary_rows[0]["sr_5"], "50.0")
        self.assertEqual(task_rows[0]["task"], "open_drawer")
        self.assertEqual(task_rows[0]["successes"], "9")
        self.assertEqual(report["source_root"], str(root))
        self.assertEqual(report["runs"][0]["success_rates"]["1"], 90.0)


if __name__ == "__main__":
    unittest.main()
