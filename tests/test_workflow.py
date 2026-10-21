import json
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from support import workspace

import experiment
import main
from contract_core import APIError, Completion, write_json


def completion(*, after=False, valid=True):
    answer = {"title": "Замечания", "steps": [f"Пункт {n} — риск — согласовать условие" if after else "Согласовать условие" for n in range(1, 6)],
              "notes": ["Предварительная проверка."]}
    return Completion(json.dumps(answer, ensure_ascii=False) if valid else "invalid", "model", "stop", {"total_tokens": 100})


class ReviewTests(unittest.TestCase):
    def test_inspect_never_reads_key_or_calls_api(self):
        with workspace() as folder:
            path = folder / "contract.txt"
            path.write_text("Личный договор", encoding="utf-8")
            with patch("main.complete") as request, patch("main.load_api_key") as key, patch("builtins.print"):
                self.assertEqual(main.review_file(path, inspect_only=True), 0)
                request.assert_not_called()
                key.assert_not_called()

    def test_review_saves_result_and_only_success_starts_dialogue(self):
        for result, status in ((completion(), 0), (completion(valid=False), 1), (APIError("Нет соединения"), 1)):
            with self.subTest(result=result), workspace() as folder:
                path = folder / "contract.txt"
                path.write_text("Личный договор и уникальный текст", encoding="utf-8")
                with patch("main.PRIVATE_RESULTS", folder / "private_contract_results"), \
                     patch("main.load_api_key", return_value="secret-key"), \
                     patch("main.complete", side_effect=[result]) as request, \
                     patch("main.run_dialogue") as chat, patch("builtins.print"):
                    self.assertEqual(main.review_file(path), status)
                report_file = next((folder / "private_contract_results").glob("*/report.json"))
                report = json.loads(report_file.read_text(encoding="utf-8"))
                self.assertNotIn("secret-key", json.dumps(report))
                self.assertNotIn("Личный договор и уникальный текст", json.dumps(report, ensure_ascii=False))
                self.assertIn("Личный договор и уникальный текст", request.call_args.args[1][-1]["content"])
                self.assertEqual(chat.call_count, 1 if status == 0 else 0)
                self.assertEqual((report_file.parent / "result.json").exists(), status == 0)

    def test_cli_modes_and_cancellation(self):
        with patch("builtins.print"), patch("main.run_experiment", return_value=0) as compare:
            self.assertEqual(main.main(["--compare"]), 0)
            compare.assert_called_once()
            self.assertEqual(main.main(["--dry-run"]), 0)
        with patch("builtins.input", return_value="q"), patch("builtins.print"):
            self.assertEqual(main.main([]), 0)
        with patch("builtins.print"), patch("main.package_project", return_value=Path("result.zip")):
            self.assertEqual(main.main(["--package"]), 0)
        with patch("main.review_file", return_value=0) as review:
            self.assertEqual(main.main(["--file", "contract.docx", "--no-chat"]), 0)
            self.assertFalse(review.call_args.kwargs["chat"])


class ExperimentTests(unittest.TestCase):
    def test_two_runs_keep_parameters_and_generate_comparison(self):
        with workspace() as folder, patch("builtins.print"), \
             patch("experiment.complete", side_effect=[completion(), completion(after=True)]) as request:
            self.assertEqual(experiment.run_experiment(key="key", root=folder), 0)
            pointer = json.loads((folder / "latest.json").read_text(encoding="utf-8"))
            result = folder / pointer["run"]
            report = json.loads((result / "report.json").read_text(encoding="utf-8"))
            self.assertTrue((result / "comparison.md").is_file())
            self.assertTrue((result / "README.md").is_file())
            self.assertEqual(request.call_args_list[0].args[1][-1], request.call_args_list[1].args[1][-1])
            self.assertTrue(all(r["validation"]["raw_valid_json"] for r in report["runs"]))

    def test_failed_rerun_preserves_previous_success(self):
        with workspace() as folder, patch("builtins.print"), \
             patch("experiment.complete", side_effect=APIError("Ошибка")) as request:
            write_json(folder / "latest.json", {"run": "previous"})
            self.assertEqual(experiment.run_experiment(key="key", root=folder), 1)
            self.assertEqual(json.loads((folder / "latest.json").read_text(encoding="utf-8")), {"run": "previous"})
            request.assert_called_once()

    def test_package_uses_allowlist_and_blocks_secret_or_bad_pointer(self):
        with workspace() as root, patch("builtins.print"), \
             patch("experiment.complete", side_effect=[completion(), completion(after=True)]), \
             patch("experiment.BASE_DIR", root), patch("experiment.RESULTS_ROOT", root / "contract_results"):
            results = root / "contract_results"
            self.assertEqual(experiment.run_experiment(key="key", root=results), 0)
            files = ["main.py", "contract_core.py", "documents.py", "dialogue.py", "experiment.py", "prompts.json",
                     "requirements.txt", "README.md", "REVIEW.md", ".env.example", ".gitignore", ".vscode/tasks.json", "tests/support.py"]
            for name in files:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("Public file", encoding="utf-8")
            (root / ".env").write_text("PROXYAPI_API_KEY=DO_NOT_EXPORT_SECRET\n", encoding="utf-8")
            private = root / "private_contract_results"
            private.mkdir()
            (private / "personal.txt").write_text("Личный договор", encoding="utf-8")
            archive = experiment.package_project()
            with ZipFile(archive) as zipped:
                self.assertFalse(any(n.startswith("private_contract_results/") or n == ".env" for n in zipped.namelist()))
                self.assertIn("tests/support.py", zipped.namelist())
            pointer = json.loads((results / "latest.json").read_text(encoding="utf-8"))
            artifact = results / pointer["run"] / "after.json"
            saved = artifact.read_bytes()
            artifact.write_text('{"incorrect": true}', encoding="utf-8")
            with self.assertRaises(ValueError):
                experiment.package_project()
            artifact.write_bytes(saved)
            (root / "README.md").write_text("DO_NOT_EXPORT_SECRET", encoding="utf-8")
            with self.assertRaises(ValueError):
                experiment.package_project()
            write_json(results / "latest.json", {"run": "../private_contract_results"})
            with self.assertRaises(ValueError):
                experiment.package_project()


if __name__ == "__main__":
    unittest.main()
