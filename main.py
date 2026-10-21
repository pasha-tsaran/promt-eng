"""Проверка договора и диалог; --compare — учебное сравнение двух промптов."""

import argparse
import json
from pathlib import Path
from typing import Any

from contract_core import (
    BASE_DIR,
    APIError,
    build_messages,
    complete,
    get_prompt,
    load_api_key,
    load_experiment_prompts,
    new_run_folder,
    now,
    parameters,
    validate_completion,
    write_checklist,
    write_json,
)
from dialogue import run_dialogue
from documents import read_contract
from experiment import package_project, run_experiment

PRIVATE_RESULTS = BASE_DIR / "private_contract_results"


def review_file(path: Path, *, inspect_only: bool = False, chat: bool = True) -> int:
    text, source = read_contract(path)
    messages = build_messages(get_prompt("file_review"), text)
    print(f"Файл: {source['file_name']}; абзацев: {source['paragraphs']}; символов: {source['characters']}", flush=True)
    if inspect_only:
        print("Файл прочитан локально. Запрос в API не отправлялся.")
        return 0
    key = load_api_key()
    folder = new_run_folder(PRIVATE_RESULTS)
    report: dict[str, Any] = {
        "executed_at": now(), **parameters(), "source": source,
        "position": "покупатель / заказчик", "system_message": messages[0]["content"],
        "status": "pending",
    }
    print(f"Первичная проверка — один запрос в ProxyAPI: {report['model']}, temperature={report['temperature']}.", flush=True)
    try:
        result = complete(key, messages)
        answer, checks = validate_completion(result)
        report.update(result.record(), validation=checks,
                      status="ok" if checks["schema_ok"] else "invalid")
        print(json.dumps(checks, ensure_ascii=False), flush=True)
        if answer is not None and checks["schema_ok"]:
            write_json(folder / "result.json", answer)
            write_checklist(folder / "README.md", answer,
                            introduction="Предварительные замечания в интересах покупателя / заказчика.")
            print(json.dumps(answer, ensure_ascii=False, indent=2), flush=True)
        else:
            print("Ответ не принят: " + " ".join(checks["errors"]))
    except APIError as exc:
        report.update(status="error", error=str(exc))
        print(str(exc), flush=True)
    write_json(folder / "report.json", report)
    print(f"Результаты: {folder}", flush=True)
    if report["status"] == "ok" and chat:
        run_dialogue(key, messages, report["raw_answer"], folder)
    return int(report["status"] != "ok")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--compare", action="store_true", help="Два платных запуска по плану урока")
    mode.add_argument("--dry-run", action="store_true", help="Проверить промпты и настройки без API")
    mode.add_argument("--package", action="store_true", help="Собрать ZIP проекта и последнего успешного сравнения")
    parser.add_argument("--file", type=Path, help="Путь к своему .docx, .txt или .md")
    parser.add_argument("--inspect", action="store_true", help="Проверить чтение документа локально, без API")
    parser.add_argument("--no-chat", action="store_true", help="Завершить после первой проверки")
    args = parser.parse_args(argv)
    if (args.compare or args.dry_run or args.package) and (args.file or args.inspect or args.no_chat):
        parser.error("--file, --inspect и --no-chat относятся только к проверке своего договора.")
    try:
        if args.dry_run:
            prompts = load_experiment_prompts()
            get_prompt("file_review")
            get_prompt("dialogue")
            print(json.dumps({**parameters(), "prompts": [p["name"] for p in prompts]}, ensure_ascii=False, indent=2))
            return 0
        if args.compare:
            return run_experiment()
        if args.package:
            print(f"Архив: {package_project()}")
            return 0
        path = args.file
        if path is None:
            print("Проверка договора в интересах покупателя / заказчика.", flush=True)
            if not args.inspect:
                print("После выбора файла текст будет отправлен в ProxyAPI. После ответа можно задавать вопросы.", flush=True)
            value = input("Полный путь к .docx/.txt/.md (q — выход): ").strip().strip('"').strip("'")
            if value.lower() in {"q", "quit", "выход"}:
                return 0
            if not value:
                raise ValueError("Путь не указан.")
            path = Path(value)
        return review_file(path.expanduser().resolve(), inspect_only=args.inspect, chat=not args.no_chat)
    except (ValueError, TypeError, OSError, APIError) as exc:
        print(f"Ошибка: {exc}")
        return 1
    except (EOFError, KeyboardInterrupt):
        print("\nРабота завершена.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
