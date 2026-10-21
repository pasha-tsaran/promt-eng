import json
import unittest
from pathlib import Path
from unittest.mock import patch

from support import workspace

import dialogue
from contract_core import APIError, Completion


def reply(reason="stop"):
    return Completion(json.dumps({"title": "Правка", "steps": ["Согласуйте возврат аванса."], "notes": []}),
                      "model", reason, {"total_tokens": 100})


class DialogueTests(unittest.TestCase):
    def setUp(self):
        self.initial = [{"role": "system", "content": "Правила"},
                        {"role": "user", "content": "Пример user"},
                        {"role": "assistant", "content": "Пример assistant"},
                        {"role": "user", "content": "Исходный договор [Абзац 1]"}]
        self.raw = json.dumps({"title": "Первая проверка", "steps": ["Риск"], "notes": []})

    def test_contract_review_and_successful_exchanges_are_preserved(self):
        with workspace() as folder, patch("builtins.input", side_effect=["Что с авансом?", "Предложи правку", "/exit"]), \
             patch("builtins.print"), patch("dialogue.complete", return_value=reply()) as request:
            dialogue.run_dialogue("secret-key", self.initial, self.raw, folder)
            log = json.loads((folder / "dialogue.json").read_text(encoding="utf-8"))
        self.assertEqual(request.call_count, 2)
        second = request.call_args.args[1]
        self.assertEqual(second[1], self.initial[-1])
        self.assertEqual(second[2]["content"], self.raw)
        self.assertEqual(second[3]["content"], "Что с авансом?")
        self.assertEqual(second[4]["content"], reply().raw_answer)
        self.assertEqual(second[5]["content"], "Предложи правку")
        self.assertIn("Отвечай прямо на последний вопрос", second[0]["content"])
        self.assertEqual(len(log["exchanges"]), 2)
        self.assertNotIn("secret-key", json.dumps(log))
        self.assertNotIn("Исходный договор", json.dumps(log, ensure_ascii=False))

    def test_failed_request_or_invalid_response_does_not_enter_context(self):
        bad = Completion("not JSON", "model", "stop", {})
        for failure in (APIError("Ошибка соединения"), bad, reply("length")):
            with self.subTest(failure=failure), workspace() as folder, \
                 patch("builtins.input", side_effect=["Неуспешный вопрос", "Новый вопрос", "/exit"]), \
                 patch("builtins.print"), patch("dialogue.complete", side_effect=[failure, reply()]) as request:
                dialogue.run_dialogue("key", self.initial, self.raw, folder)
                log = json.loads((folder / "dialogue.json").read_text(encoding="utf-8"))
            self.assertEqual(len(request.call_args.args[1]), 4)
            self.assertNotEqual(log["exchanges"][0]["status"], "ok")

    def test_commands_and_empty_input_do_not_add_api_requests(self):
        with workspace() as folder, \
             patch("builtins.input", side_effect=["", "/help", "Первый", "/reset", "Второй", "выход"]), \
             patch("builtins.print"), patch("dialogue.complete", return_value=reply()) as request:
            dialogue.run_dialogue("key", self.initial, self.raw, folder)
            log = json.loads((folder / "dialogue.json").read_text(encoding="utf-8"))
        self.assertEqual(request.call_count, 2)
        self.assertEqual(len(request.call_args.args[1]), 4)
        self.assertEqual(log["exchanges"][1]["event"], "reset")

    def test_count_and_character_budget_drop_only_complete_old_pairs(self):
        pinned = [self.initial[-1], {"role": "assistant", "content": self.raw}]
        recent = [{"role": role, "content": label * 50} for label in ("A", "B", "C") for role in ("user", "assistant")]
        base = dialogue.dialogue_messages(pinned, [], "Q")
        size = sum(len(m["content"]) for m in base)
        with patch("dialogue.MAX_CONTEXT_CHARS", size + 100):
            messages = dialogue.dialogue_messages(pinned, recent, "Q")
        self.assertEqual(messages[1:3], pinned)
        self.assertEqual(messages[3:-1], recent[-2:])
        with patch("dialogue.MAX_HISTORY_TURNS", 1):
            self.assertEqual(dialogue.dialogue_messages(pinned, recent, "Q")[3:-1], recent[-2:])
        with patch("dialogue.MAX_CONTEXT_CHARS", 1), self.assertRaises(ValueError):
            dialogue.dialogue_messages(pinned, recent, "Q")

    def test_exit_eof_and_overlong_question_do_not_call_api(self):
        for finish in (EOFError(), KeyboardInterrupt(), "q"):
            with self.subTest(finish=finish), patch("builtins.input", side_effect=[finish]), \
                 patch("builtins.print"), patch("dialogue.complete") as request:
                dialogue.run_dialogue("key", self.initial, self.raw, Path("unused"))
            request.assert_not_called()
        with patch("builtins.input", side_effect=["x" * 12_001, "/exit"]), \
             patch("builtins.print"), patch("dialogue.complete") as request:
            dialogue.run_dialogue("key", self.initial, self.raw, Path("unused"))
        request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
