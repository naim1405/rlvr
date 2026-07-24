import re


class ArithmeticVerifier:
    """Verifier for P1: Arithmetic tasks."""

    def __init__(self, reward_params: dict = None):
        reward_params = reward_params or {}
        self.tolerance = float(reward_params.get("tolerance", 1e-5))

    def extract_number(self, text: str) -> float | None:
        """Extract floating point number from model response."""
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        ans_str = match.group(1).strip() if match else text.strip()
        
        # Look for numbers (including decimals, fractions, or negative numbers)
        num_match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", ans_str)
        if num_match:
            try:
                return float(num_match.group(0))
            except ValueError:
                return None
        return None

    def verify(self, model_response: str, extra_env_info: dict) -> float:
        """Verifies numerical result against expected value within tolerance."""
        expected_raw = extra_env_info.get("expected_answer", None)
        if expected_raw is None:
            return 0.0

        try:
            expected = float(expected_raw)
        except (ValueError, TypeError):
            return 0.0

        predicted = self.extract_number(model_response)
        if predicted is None:
            return 0.0

        if abs(predicted - expected) <= self.tolerance:
            return 1.0
        return 0.0
