import copy
import json
import unittest
from unittest.mock import Mock, patch

import requests
from support import workspace

import contract_core as core


def answer(steps=1):
    return {"title": "Проверка", "steps": ["Согласовать условие"] * steps, "notes": ["Предварительно"]}


class JSONTests(unittest.TestCase):
    def test_schema_and_raw_json_are_checked_without_repairs(self):
        raw = json.dumps(answer())
        self.assertTrue(core.parse_answer(raw)[1]["schema_ok"])
        for invalid in (f"```json\n{raw}\n```", f"Ответ: {raw}", '{"x":NaN}', '{"x":1,"x":2}'):
            with self.subTest(raw=invalid):
                self.assertFalse(core.parse_answer(invalid)[1]["raw_valid_json"])
        bad = [[], {**answer(), "extra": 1}, {**answer(), "title": " "},
               {**answer(), "steps": []}, {**answer(), "steps": [1]}, {**answer(), "steps": [" "]},
               {**answer(), "steps": ["x"] * 9}, {**answer(), "notes": "x"},
               {**answer(), "notes": ["n"] * 4}, {**answer(), "notes": ["word " * 26]}]
        for value in bad:
            with self.subTest(value=value):
                self.assertFalse(core.parse_answer(json.dumps(value))[1]["schema_ok"])
        self.assertTrue(core.parse_answer(json.dumps({**answer(), "notes": []}))[1]["schema_ok"])
        self.assertFalse(core.parse_answer(raw, min_steps=5)[1]["schema_ok"])

    def test_nonstandard_finish_reason_is_not_accepted(self):
        for reason in ("length", "content_filter", "tool_calls", None):
            result = core.Completion(json.dumps(answer()), "model", reason, {})
            self.assertFalse(core.validate_completion(result)[1]["schema_ok"])

    def test_atomic_write_and_readable_artifact(self):
        with workspace() as folder:
            path = folder / "result.json"
            core.write_json(path, answer())
            with self.assertRaises(TypeError):
                core.write_json(path, {"invalid": object()})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), answer())
            self.assertFalse(list(folder.glob("*.tmp")))
            core.write_checklist(folder / "README.md", answer(), introduction="Проверка покупателя")
            self.assertIn(answer()["steps"][0], (folder / "README.md").read_text(encoding="utf-8"))


class ClientTests(unittest.TestCase):
    def response(self, data):
        response = Mock()
        response.json.return_value = data
        return response

    def test_payload_metadata_and_no_key_in_result(self):
        data = {"choices": [{"message": {"content": json.dumps(answer())}, "finish_reason": "stop"}],
                "usage": {"total_tokens": 100}, "model": core.MODEL}
        with patch("contract_core.requests.post", return_value=self.response(data)) as post:
            result = core.complete("private-key", [{"role": "user", "content": "текст"}])
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], core.MODEL)
        self.assertEqual(payload["temperature"], 0.2)
        self.assertTrue(payload["response_format"]["json_schema"]["strict"])
        self.assertNotIn("private-key", json.dumps(result.record()))
        self.assertEqual(result.usage["total_tokens"], 100)

    def test_transport_errors_are_safe_and_not_retried(self):
        http = requests.HTTPError("secret in server message")
        http.response = Mock(status_code=401)
        for error in (requests.Timeout("secret"), requests.ConnectionError("secret"), http):
            with self.subTest(error=error), patch("contract_core.requests.post", side_effect=error) as post:
                with self.assertRaises(core.APIError) as caught:
                    core.complete("secret", [])
                self.assertNotIn("secret", str(caught.exception))
                post.assert_called_once()

    def test_malformed_or_refused_responses_are_safe(self):
        cases = [[], {}, {"choices": []}, {"choices": [1]}, {"choices": [{}]},
                 {"choices": [{"message": {"content": ""}}]},
                 {"choices": [{"message": {"refusal": "refused"}}]}]
        for data in cases:
            with self.subTest(data=data), patch("contract_core.requests.post", return_value=self.response(data)), \
                 self.assertRaises(core.APIError):
                core.complete("key", [])
        response = self.response({})
        response.json.side_effect = ValueError("bad JSON")
        with patch("contract_core.requests.post", return_value=response), self.assertRaises(core.APIError):
            core.complete("key", [])

    def test_context_limit_prevents_network_request(self):
        with patch("contract_core.MAX_CONTEXT_CHARS", 5), patch("contract_core.requests.post") as post:
            with self.assertRaises(core.APIError):
                core.complete("key", [{"role": "user", "content": "too long"}])
            post.assert_not_called()

    def test_key_loading_and_placeholder(self):
        with patch("contract_core.dotenv_values", return_value={"PROXYAPI_API_KEY": "file-key"}), \
             patch("contract_core.os.getenv", return_value="environment-key"):
            self.assertEqual(core.load_api_key(), "environment-key")
        with patch("contract_core.dotenv_values", return_value={"PROXYAPI_API_KEY": "your_key"}), \
             patch("contract_core.os.getenv", return_value=None), self.assertRaises(ValueError):
            core.load_api_key()


class PromptTests(unittest.TestCase):
    def test_only_role_iteration_is_allowed(self):
        prompts = core.load_experiment_prompts()
        before, after = prompts
        self.assertNotEqual(before["role"], after["role"])
        for field in ("context", "question", "format", "test_input", "few_shot"):
            self.assertEqual(before[field], after[field])
        self.assertEqual(core.build_messages(before)[-1], core.build_messages(after)[-1])

    def test_few_shot_iteration_is_also_allowed(self):
        prompts = copy.deepcopy(core.load_experiment_prompts())
        prompts[1]["role"] = prompts[0]["role"]
        prompts[1]["few_shot"] = [{"role": "user", "content": "вопрос"}, {"role": "assistant", "content": "ответ"}]
        self.assertEqual(len(core.load_experiment_prompts({"prompts": prompts})), 2)
        self.assertEqual([m["role"] for m in core.build_messages(prompts[1])], ["system", "user", "assistant", "user"])

    def test_invalid_iteration_and_prompt_are_rejected(self):
        prompts = copy.deepcopy(core.load_experiment_prompts())
        prompts[1]["test_input"] = "другой договор"
        with self.assertRaises(ValueError):
            core.load_experiment_prompts({"prompts": prompts})
        for invalid in (None, {}, {**prompts[0], "few_shot": [{}]},
                        {**prompts[0], "few_shot": [{"role": "system"}, {"role": "assistant"}]}):
            with self.subTest(invalid=invalid), self.assertRaises((ValueError, TypeError)):
                core.validate_prompt(invalid)


if __name__ == "__main__":
    unittest.main()
