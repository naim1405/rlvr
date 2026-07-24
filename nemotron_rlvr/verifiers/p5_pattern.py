import re


class PatternVerifier:
    """Verifier for P5: Pattern / Inductive Reasoning tasks."""

    def __init__(self, reward_params: dict = None):
        reward_params = reward_params or {}

    def extract_pattern_answer(self, text: str) -> str:
        """Extract sequence continuation answer."""
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        ans = match.group(1).strip() if match else text.strip()
        return ans.lower()

    def verify(self, model_response: str, extra_env_info: dict) -> float:
        """Verifies pattern completion against expected continuation."""
        expected = str(extra_env_info.get("expected_answer", "")).strip().lower()
        predicted = self.extract_pattern_answer(model_response)

        if not predicted or not expected:
            return 0.0

        if predicted == expected or expected in predicted:
            return 1.0
        return 0.0
