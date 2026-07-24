import re


class LogicVerifier:
    """Verifier for P3: Deductive Logic tasks."""

    def __init__(self, reward_params: dict = None):
        reward_params = reward_params or {}

    def extract_bool(self, text: str) -> bool | str | None:
        """Extract truth value (True/False, Yes/No, Valid/Invalid)."""
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        ans_str = match.group(1).strip().lower() if match else text.strip().lower()

        if "true" in ans_str or "valid" in ans_str or "yes" in ans_str:
            return "true"
        if "false" in ans_str or "invalid" in ans_str or "no" in ans_str:
            return "false"
        return ans_str

    def verify(self, model_response: str, extra_env_info: dict) -> float:
        """Verifies logical truth value against expected solution."""
        expected = str(extra_env_info.get("expected_answer", "")).strip().lower()
        predicted = self.extract_bool(model_response)

        if predicted is None:
            return 0.0

        if predicted == expected or expected in str(predicted):
            return 1.0
        return 0.0
