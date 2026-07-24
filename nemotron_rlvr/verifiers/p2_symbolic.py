import re


class SymbolicVerifier:
    """Verifier for P2: Symbolic Transformation tasks."""

    def __init__(self, reward_params: dict = None):
        reward_params = reward_params or {}
        self.ignore_spaces = reward_params.get("ignore_spaces", True)

    def normalize(self, expr: str) -> str:
        """Clean and normalize symbolic expression string."""
        if self.ignore_spaces:
            expr = re.sub(r"\s+", "", expr)
        return expr.strip().lower()

    def extract_expression(self, text: str) -> str:
        """Extract expression inside <answer> tags or return response."""
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else text.strip()

    def verify(self, model_response: str, extra_env_info: dict) -> float:
        """Verifies symbolic output string against target expression."""
        expected = str(extra_env_info.get("expected_answer", ""))
        predicted = self.extract_expression(model_response)

        if not predicted or not expected:
            return 0.0

        norm_predicted = self.normalize(predicted)
        norm_expected = self.normalize(expected)

        if norm_predicted == norm_expected:
            return 1.0
        return 0.0
