import re


class PlanningVerifier:
    """Verifier for P4: Procedural / Planning tasks."""

    def __init__(self, reward_params: dict = None):
        reward_params = reward_params or {}
        self.allow_partial_credit = reward_params.get("allow_partial_credit", True)

    def extract_steps(self, text: str) -> list[str]:
        """Extract ordered plan steps from response."""
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        content = match.group(1) if match else text
        lines = [line.strip() for line in content.split("\n") if line.strip()]
        return lines

    def verify(self, model_response: str, extra_env_info: dict) -> float:
        """Verifies step-by-step plan against expected procedural steps."""
        expected_steps = extra_env_info.get("expected_steps", [])
        if not expected_steps and "expected_answer" in extra_env_info:
            expected_steps = [str(extra_env_info["expected_answer"])]

        predicted_steps = self.extract_steps(model_response)
        if not predicted_steps or not expected_steps:
            return 0.0

        matches = 0
        for p_step, e_step in zip(predicted_steps, expected_steps):
            if e_step.lower() in p_step.lower():
                matches += 1

        if matches == len(expected_steps):
            return 1.0

        if self.allow_partial_credit and len(expected_steps) > 0:
            return matches / len(expected_steps)

        return 0.0
