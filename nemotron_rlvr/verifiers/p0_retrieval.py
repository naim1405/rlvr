import re


class RetrievalVerifier:
    """Verifier for P0: Factual Retrieval tasks."""

    def __init__(self, reward_params: dict = None):
        reward_params = reward_params or {}
        self.case_sensitive = reward_params.get("case_sensitive", False)
        self.strip_punctuation = reward_params.get("strip_punctuation", True)

    def extract_answer(self, text: str) -> str:
        """Extract answer content from <answer> tags or return raw text."""
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        ans = match.group(1).strip() if match else text.strip()
        if self.strip_punctuation:
            ans = re.sub(r"[^\w\s]", "", ans)
        if not self.case_sensitive:
            ans = ans.lower()
        return ans

    def verify(self, model_response: str, extra_env_info: dict) -> float:
        """Verifies model response against expected ground truth string."""
        expected = str(extra_env_info.get("expected_answer", "")).strip()
        if self.strip_punctuation:
            expected = re.sub(r"[^\w\s]", "", expected)
        if not self.case_sensitive:
            expected = expected.lower()

        predicted = self.extract_answer(model_response)

        if not predicted or not expected:
            return 0.0

        if predicted == expected or expected in predicted:
            return 1.0
        return 0.0
