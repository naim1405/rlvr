import re


class ConstraintVerifier:
    """Verifier for P6: Constraint Satisfaction tasks."""

    def __init__(self, reward_params: dict = None):
        reward_params = reward_params or {}
        self.allow_partial_credit = reward_params.get("allow_partial_credit", True)

    def extract_solution(self, text: str) -> str:
        """Extract solution inside <answer> tags."""
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else text.strip()

    def verify(self, model_response: str, extra_env_info: dict) -> float:
        """Verifies solution against specified constraint set or expected solution."""
        expected = str(extra_env_info.get("expected_answer", "")).strip().lower()
        constraints = extra_env_info.get("constraints", [])
        predicted = self.extract_solution(model_response).lower()

        if not predicted:
            return 0.0

        if expected and predicted == expected:
            return 1.0

        if constraints:
            satisfied = 0
            for c in constraints:
                if str(c).lower() in predicted:
                    satisfied += 1
            if satisfied == len(constraints):
                return 1.0
            if self.allow_partial_credit and len(constraints) > 0:
                return satisfied / len(constraints)

        if expected and expected in predicted:
            return 1.0

        return 0.0
