"""Уточняющие вопросы с сохранением исходного договора и первой проверки."""

import json
from pathlib import Path
from typing import Any

from contract_core import (
    MAX_CONTEXT_CHARS,
    MAX_HISTORY_TURNS,
    APIError,
    complete,
    get_prompt,
    now,
    parameters,
    validate_completion,
    write_json,
)

MAX_QUESTION_CHARS = 12_000


def dialogue_messages(
    pinned: list[dict[str, str]], recent: list[dict[str, str]], question: str,
) -> list[dict[str, str]]:
    prompt = get_prompt("dialogue")
    system = {"role": "system", "content":
              f"Роль:\n{prompt['role']}\n\nКонтекст:\n{prompt['context']}\n\n"
              f"Задача:\n{prompt['question']}\n\nФормат ответа:\n{prompt['format']}"}
    history = list(recent[-MAX_HISTORY_TURNS * 2:])
    current = {"role": "user", "content": question}
    while history and sum(len(m["content"]) for m in [system, *pinned, *history, current]) > MAX_CONTEXT_CHARS:
        del history[:2]
    messages = [system, *pinned, *history, current]
    if sum(len(m["content"]) for m in messages) > MAX_CONTEXT_CHARS:
        raise ValueError("Договор и вопрос превышают лимит контекста. Сократите вопрос или разделите договор.")
    return messages


def run_dialogue(key: str, initial_messages: list[dict[str, str]], initial_raw: str, folder: Path) -> None:
    pinned = [dict(initial_messages[-1]), {"role": "assistant", "content": initial_raw}]
    recent: list[dict[str, str]] = []
    log: dict[str, Any] = {
        **parameters(), "max_recent_exchanges": MAX_HISTORY_TURNS,
        "max_context_characters": MAX_CONTEXT_CHARS,
        "context_policy": "Договор и первая проверка закреплены; затем до 10 последних успешных пар в пределах лимита символов.",
        "exchanges": [],
    }
    print("\nВаши вопросы по договору. Каждый вопрос — отдельный платный запрос.", flush=True)
    print("/exit — выход; /reset — очистить уточнения; /help — команды.", flush=True)
    while True:
        try:
            question = input("\nВаш вопрос: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДиалог завершен.")
            return
        command = question.lower()
        if command in {"/exit", "выход", "exit", "quit", "q", "стоп"}:
            print("Диалог завершен.")
            return
        if command == "/help":
            print("Задайте вопрос или попросите правку. /exit — выход; /reset — очистить уточнения, сохранив договор.")
            continue
        if command == "/reset":
            recent.clear()
            log["exchanges"].append({"event": "reset", "at": now()})
            write_json(folder / "dialogue.json", log)
            print("Уточнения очищены из контекста. Договор и первая проверка сохранены.")
            continue
        if not question or len(question) > MAX_QUESTION_CHARS:
            print(f"Введите непустой вопрос до {MAX_QUESTION_CHARS} символов или /exit.")
            continue
        try:
            messages = dialogue_messages(pinned, recent, question)
        except ValueError as exc:
            print(str(exc))
            continue
        entry: dict[str, Any] = {"question": question, "asked_at": now(),
                                 "retained_exchanges": (len(messages) - 4) // 2}
        print("Ожидание ответа…", flush=True)
        try:
            result = complete(key, messages)
            answer, checks = validate_completion(result)
            entry.update(result.record(), validation=checks,
                         status="ok" if checks["schema_ok"] else "invalid")
            if answer is not None and checks["schema_ok"]:
                recent = messages[3:-1] + [messages[-1], {"role": "assistant", "content": result.raw_answer}]
                recent = recent[-MAX_HISTORY_TURNS * 2:]
                print(json.dumps(answer, ensure_ascii=False, indent=2), flush=True)
            else:
                print("Ответ не принят: " + " ".join(checks["errors"]) + " Контекст не изменен.")
        except APIError as exc:
            entry.update(status="error", error=str(exc))
            print(f"{exc} Можно повторить вопрос; контекст не изменен.")
        log["exchanges"].append(entry)
        write_json(folder / "dialogue.json", log)
