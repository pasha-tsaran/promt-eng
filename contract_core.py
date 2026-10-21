"""Общие настройки, промпты, клиент ProxyAPI и проверка результата."""

import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import requests
from dotenv import dotenv_values

BASE_DIR = Path(__file__).resolve().parent
API_URL = "https://api.proxyapi.ru/v1/chat/completions"
MODEL = "anthropic/claude-haiku-4-5"
TEMPERATURE = 0.2
MAX_TOKENS = 3000
MAX_HISTORY_TURNS = 10
MAX_CONTEXT_CHARS = 180_000
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "contract_review",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Непустой заголовок на русском языке."},
                "steps": {
                    "type": "array", "items": {"type": "string"},
                    "description": "От 1 до 8 непустых смысловых пунктов, соответствующих задаче.",
                },
                "notes": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Не более 3 коротких примечаний, до 25 слов каждое.",
                },
            },
            "required": ["title", "steps", "notes"],
            "additionalProperties": False,
        },
    },
}


class APIError(Exception):
    """Сообщение, которое можно показать без раскрытия ключа и HTTP-заголовков."""


@dataclass(frozen=True)
class Completion:
    raw_answer: str
    response_model: str | None
    finish_reason: str | None
    usage: dict[str, Any]

    def record(self) -> dict[str, Any]:
        return asdict(self)


def now() -> str:
    return datetime.now(timezone(timedelta(hours=3))).isoformat()


def parameters() -> dict[str, Any]:
    return {"api_url": API_URL, "model": MODEL, "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS, "response_format": RESPONSE_FORMAT}


def load_api_key() -> str:
    values = dotenv_values(BASE_DIR / ".env", encoding="utf-8-sig", interpolate=False)
    key = (os.getenv("PROXYAPI_API_KEY") or values.get("PROXYAPI_API_KEY") or "").strip()
    if not key or key.startswith("your_"):
        raise ValueError("Укажите PROXYAPI_API_KEY в .env рядом с main.py.")
    return key


def complete(key: str, messages: list[dict[str, str]]) -> Completion:
    """Один запрос без автоматических повторов; формат общий для всех режимов."""
    if sum(len(message["content"]) for message in messages) > MAX_CONTEXT_CHARS:
        raise APIError("Запрос превышает лимит контекста приложения. Разделите документ или сократите вопрос.")
    try:
        response = requests.post(
            API_URL, headers={"Authorization": f"Bearer {key}"},
            json={"model": MODEL, "messages": messages, "temperature": TEMPERATURE,
                  "max_tokens": MAX_TOKENS, "response_format": RESPONSE_FORMAT},
            timeout=(10, 120),
        )
        response.raise_for_status()
    except requests.Timeout:
        raise APIError("Истекло время ожидания ProxyAPI.") from None
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else 0
        hints = {401: "Проверьте API-ключ.", 402: "Проверьте баланс ProxyAPI.",
                 403: "Нет доступа к модели.", 429: "Достигнут лимит запросов."}
        raise APIError(f"Ошибка ProxyAPI, HTTP {status}. " + hints.get(status, "Проверьте доступность сервиса.")) from None
    except requests.RequestException:
        raise APIError("Не удалось соединиться с ProxyAPI.") from None
    try:
        data = response.json()
    except ValueError:
        raise APIError("ProxyAPI вернул невалидный JSON HTTP-ответа.") from None
    if not isinstance(data, dict):
        raise APIError("ProxyAPI вернул ответ неожиданной структуры.")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise APIError("ProxyAPI не вернул вариант ответа.")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise APIError("ProxyAPI не вернул сообщение модели.")
    if message.get("refusal"):
        raise APIError("Модель отказалась обрабатывать запрос.")
    raw = message.get("content")
    if not isinstance(raw, str) or not raw.strip():
        raise APIError("Модель не вернула текст ответа.")
    usage = data.get("usage")
    model = data.get("model")
    finish = choice.get("finish_reason")
    return Completion(raw, model if isinstance(model, str) else None,
                      finish if isinstance(finish, str) else None,
                      usage if isinstance(usage, dict) else {})


def _reject_constant(value: str) -> None:
    raise ValueError(f"Недопустимая константа JSON: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Повторяющийся ключ JSON: {key}")
        result[key] = value
    return result


def parse_answer(raw: str, *, min_steps: int = 1) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Проверяет исходный ответ, не удаляет Markdown и не исправляет данные."""
    checks: dict[str, Any] = {"valid_json": False, "raw_valid_json": False,
                              "schema_ok": False, "errors": []}
    try:
        answer = json.loads(raw, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
    except ValueError:
        checks["errors"].append("Ответ не является чистым JSON.")
        return None, checks
    checks.update(valid_json=True, raw_valid_json=True)
    if not isinstance(answer, dict) or set(answer) != {"title", "steps", "notes"}:
        checks["errors"].append("Нужен объект с полями title, steps, notes и без других полей.")
        return None, checks
    if not isinstance(answer["title"], str) or not answer["title"].strip():
        checks["errors"].append("title должен быть непустой строкой.")
    for field in ("steps", "notes"):
        if not isinstance(answer[field], list) or not all(isinstance(x, str) and x.strip() for x in answer[field]):
            checks["errors"].append(f"{field} должен быть массивом непустых строк.")
    if checks["errors"]:
        return None, checks
    steps, notes = answer["steps"], answer["notes"]
    if not min_steps <= len(steps) <= 8:
        checks["errors"].append(f"Нужно от {min_steps} до 8 пунктов steps.")
    if len(notes) > 3 or any(len(x.split()) > 25 for x in notes):
        checks["errors"].append("Допустимо до 3 примечаний, до 25 слов каждое.")
    checks.update(schema_ok=not checks["errors"], steps_count=len(steps), notes_count=len(notes),
                  notes_word_counts=[len(x.split()) for x in notes],
                  steps_with_clause_reference=sum(bool(re.search(r"[Пп]ункт\s+\d", s)) for s in steps))
    return answer, checks


def validate_completion(result: Completion, *, min_steps: int = 1) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    answer, checks = parse_answer(result.raw_answer, min_steps=min_steps)
    if result.finish_reason != "stop":
        checks["errors"].append("Ответ не завершен штатно: " + (result.finish_reason or "причина не указана"))
        checks["schema_ok"] = False
    return answer, checks


def load_catalog(path: Path | None = None) -> dict[str, Any]:
    data = json.loads((path or BASE_DIR / "prompts.json").read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("Корень prompts.json должен быть объектом.")
    return data


def validate_prompt(prompt: Any) -> dict[str, Any]:
    if not isinstance(prompt, dict):
        raise TypeError("Промпт должен быть объектом.")
    for field in ("name", "role", "context", "question", "format", "test_input"):
        if not isinstance(prompt.get(field), str) or not prompt[field].strip():
            raise ValueError(f"Не заполнено поле промпта: {field}.")
    examples = prompt.get("few_shot", [])
    if not isinstance(examples, list) or len(examples) % 2:
        raise ValueError("few_shot должен содержать полные пары user/assistant.")
    for index, example in enumerate(examples):
        if not isinstance(example, dict) or example.get("role") != ("user" if index % 2 == 0 else "assistant"):
            raise ValueError("Нарушен порядок ролей в few_shot.")
        if not isinstance(example.get("content"), str) or not example["content"].strip():
            raise ValueError("Пустое сообщение few_shot.")
    return prompt


def load_experiment_prompts(catalog: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    prompts = (catalog if catalog is not None else load_catalog()).get("prompts")
    if not isinstance(prompts, list) or len(prompts) != 2:
        raise ValueError("Для задания нужны два промпта: до и после улучшения.")
    for prompt in prompts:
        validate_prompt(prompt)
    before, after = prompts
    changed = {key for key in before.keys() | after.keys() if before.get(key) != after.get(key)}
    if changed - {"id", "name", "role", "few_shot"} or not changed & {"role", "few_shot"}:
        raise ValueError("Улучшайте роль или few_shot; контекст, задачу, формат и договор оставьте одинаковыми.")
    if [p.get("id") for p in prompts] != [1, 2]:
        raise ValueError("Ожидаются id 1 и 2.")
    return prompts


def get_prompt(name: str) -> dict[str, Any]:
    return validate_prompt(load_catalog().get(name))


def build_messages(prompt: dict[str, Any], text: str | None = None) -> list[dict[str, str]]:
    validate_prompt(prompt)
    return [
        {"role": "system", "content": f"Роль:\n{prompt['role']}\n\nФормат ответа:\n{prompt['format']}"},
        *[dict(message) for message in prompt.get("few_shot", [])],
        {"role": "user", "content": f"Контекст:\n{prompt['context']}\n\nЗадача:\n{prompt['question']}"
         f"\n\nНачало текста договора:\n{text if text is not None else prompt['test_input']}\nКонец текста договора."},
    ]


def new_run_folder(root: Path) -> Path:
    stamp = datetime.now(timezone(timedelta(hours=3))).strftime("%Y%m%d_%H%M%S_%f")
    folder = root / stamp
    folder.mkdir(parents=True)
    return folder


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_checklist(path: Path, answer: dict[str, Any], *, introduction: str) -> None:
    lines = [f"# {answer['title']}", "", introduction, "", "## Действия", ""]
    lines.extend(f"{number}. {step}" for number, step in enumerate(answer["steps"], 1))
    if answer["notes"]:
        lines += ["", "## Примечания", ""]
        lines.extend(f"- {note}" for note in answer["notes"])
    lines += ["", "Это предварительная проверка по тексту, не исчерпывающая юридическая экспертиза.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
