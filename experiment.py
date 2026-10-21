"""Два запуска по плану урока, сравнение и архив для сдачи."""

import json
import re
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from dotenv import dotenv_values

from contract_core import (
    BASE_DIR,
    APIError,
    Completion,
    build_messages,
    complete,
    load_api_key,
    load_experiment_prompts,
    new_run_folder,
    now,
    parameters,
    validate_completion,
    write_checklist,
    write_json,
)

RESULTS_ROOT = BASE_DIR / "contract_results"


def write_comparison(folder: Path, report: dict[str, Any]) -> None:
    before, after = report["runs"]
    b, a = before["validation"], after["validation"]

    def notes(checks: dict[str, Any]) -> str:
        return f"{checks['notes_count']} тезиса; слов: " + ", ".join(map(str, checks["notes_word_counts"]))

    def useful(checks: dict[str, Any]) -> str:
        return f"{checks['steps_count']} действий, из них {checks['steps_with_clause_reference']} с номером пункта"

    lines = ["# Сравнение промптов до и после улучшения", "",
             f"Начало запусков: {report['executed_at']}.", "",
             "Одинаковые модель, температура, лимит ответа, JSON Schema и тестовый договор.",
             report["iteration"], "",
             "| Критерий | До | После |", "| --- | --- | --- |",
             f"| Валидный исходный JSON | {'Да' if b['raw_valid_json'] else 'Нет'} | {'Да' if a['raw_valid_json'] else 'Нет'} |",
             f"| Практичность steps: привязка к тексту | {useful(b)} | {useful(a)} |",
             f"| Лаконичность notes | {notes(b)} | {notes(a)} |", "",
             "Содержание действий: [до](before.json), [после](after.json).",
             "Номера пунктов помогают сопоставлять замечания с договором, но сами по себе не доказывают качество анализа.",
             "Одна пара запусков не доказывает устойчивость на всех документах.", ""]
    (folder / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


def run_experiment(*, key: str | None = None, root: Path | None = None) -> int:
    prompts = load_experiment_prompts()
    key = key or load_api_key()
    root = root or RESULTS_ROOT
    folder = new_run_folder(root)
    changes = []
    if prompts[0]["role"] != prompts[1]["role"]:
        changes.append("уточнена роль")
    if prompts[0].get("few_shot", []) != prompts[1].get("few_shot", []):
        changes.append("добавлен или изменен мини-пример")
    report: dict[str, Any] = {
        "kind": "lesson_experiment", "executed_at": now(), **parameters(),
        "iteration": "; ".join(changes).capitalize() + ". Тестовый договор, контекст, задача и формат одинаковы.",
        "postprocessing": "Исходный JSON проверяется без исправления и удаления Markdown.",
        "runs": [],
    }
    for label, prompt in zip(("before", "after"), prompts):
        messages = build_messages(prompt)
        row: dict[str, Any] = {"version": label, "prompt_id": prompt["id"],
                               "name": prompt["name"], "messages": messages}
        print(f"Запуск {label}: {report['model']}, temperature={report['temperature']}, max_tokens={report['max_tokens']}", flush=True)
        try:
            result = complete(key, messages)
            answer, checks = validate_completion(result, min_steps=5)
            row.update(result.record(), validation=checks, answer=answer,
                       status="ok" if checks["schema_ok"] else "invalid")
            print(json.dumps(checks, ensure_ascii=False), flush=True)
            if answer is not None and checks["schema_ok"]:
                write_json(folder / f"{label}.json", answer)
                if label == "after":
                    write_checklist(folder / "README.md", answer,
                                    introduction="Чек-лист по учебному договору № У-17 со стороны покупателя.")
        except APIError as exc:
            row.update(status="error", error=str(exc))
            print(str(exc), flush=True)
        report["runs"].append(row)
        write_json(folder / "report.json", report)
        if row["status"] == "error":
            break
    success = len(report["runs"]) == 2 and all(row["status"] == "ok" for row in report["runs"])
    if success:
        write_comparison(folder, report)
        write_json(root / "latest.json", {"run": folder.name})
        (root / "README.md").write_text(
            f"# Последнее успешное сравнение\n\n[Сравнение ответов]({folder.name}/comparison.md)\n\n"
            f"[Пользовательский чек-лист]({folder.name}/README.md)\n\n"
            f"[Параметры и исходные ответы]({folder.name}/report.json)\n", encoding="utf-8")
    print(f"Результаты: {folder}", flush=True)
    return int(not success)


def package_project(target: Path | None = None) -> Path:
    target = target or BASE_DIR / "contract_review_homework.zip"
    pointer = json.loads((RESULTS_ROOT / "latest.json").read_text(encoding="utf-8"))
    run = pointer.get("run", "")
    if not isinstance(run, str) or not re.fullmatch(r"\d{8}_\d{6}_\d{6}", run):
        raise ValueError("Неверная ссылка на результаты сравнения.")
    folder = RESULTS_ROOT / run
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    if report.get("kind") != "lesson_experiment" or len(report.get("runs", [])) != 2:
        raise ValueError("Для архива сначала выполните успешное учебное сравнение.")
    if any(report.get(name) != value for name, value in parameters().items()):
        raise ValueError("Настройки сравнения отличаются от текущих. Выполните --compare.")
    for row, prompt in zip(report["runs"], load_experiment_prompts()):
        if row.get("status") != "ok" or row.get("messages") != build_messages(prompt):
            raise ValueError("Результаты не соответствуют текущим учебным промптам. Выполните --compare.")
        raw = row.get("raw_answer")
        if not isinstance(raw, str):
            raise TypeError("В отчете отсутствует строка исходного ответа модели.")
        answer, checks = validate_completion(Completion(raw, None, row.get("finish_reason"), {}), min_steps=5)
        artifact = json.loads((folder / f"{row['version']}.json").read_text(encoding="utf-8"))
        if not checks["schema_ok"] or answer != row.get("answer") or artifact != answer:
            raise ValueError("Артефакты сравнения не соответствуют исходным ответам модели.")
    names = ["main.py", "contract_core.py", "documents.py", "dialogue.py", "experiment.py", "prompts.json",
             "requirements.txt", "README.md", "REVIEW.md", ".env.example", ".gitignore", ".vscode/tasks.json",
             "contract_results/README.md", "contract_results/latest.json"]
    files = [BASE_DIR / name for name in names]
    files += sorted((BASE_DIR / "tests").glob("test_*.py"))
    files.append(BASE_DIR / "tests" / "support.py")
    files += [folder / name for name in ("report.json", "before.json", "after.json", "comparison.md", "README.md")]
    values = dotenv_values(BASE_DIR / ".env", interpolate=False, encoding="utf-8-sig")
    secrets = [value.encode() for value in values.values() if value and not value.startswith("your_")]
    contents = [(file.relative_to(BASE_DIR).as_posix(), file.read_bytes()) for file in files]
    if any(secret in content for _, content in contents for secret in secrets):
        raise ValueError("Архив не создан: в выбранных файлах найден секрет из .env.")
    with ZipFile(target, "w", ZIP_DEFLATED) as archive:
        for name, content in contents:
            archive.writestr(name, content)
    return target
