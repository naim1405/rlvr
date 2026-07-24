import re


class PlanningVerifier:
    """Verifier for P4: Procedural / Planning tasks (PDDL State Transition Simulator)."""

    def __init__(self, reward_params: dict = None):
        reward_params = reward_params or {}
        self.allow_partial_credit = reward_params.get("allow_partial_credit", True)

    def extract_steps(self, text: str) -> list[str]:
        """Extract ordered plan steps from response inside <answer> tags."""
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        content = match.group(1) if match else text
        lines = [line.strip() for line in content.split("\n") if line.strip()]
        return lines

    def simulate_pddl(self, steps: list[str], meta: dict) -> float:
        """Simulate PDDL state transitions from initial facts to goal state."""
        init_facts = set(meta.get("init", []))
        goal_facts = set(meta.get("goal", []))
        actions = meta.get("actions", {})

        if not init_facts or not goal_facts or not actions:
            return 0.0

        current_state = set(init_facts)

        for step in steps:
            step_clean = step.strip()
            if step_clean not in actions:
                continue

            action_def = actions[step_clean]
            preconds = set(action_def.get("pre", []))
            add_effects = set(action_def.get("add", []))
            del_effects = set(action_def.get("del", []))

            # Preconditions check
            if not preconds.issubset(current_state):
                return 0.0  # Invalid action execution

            # State transition
            current_state.difference_update(del_effects)
            current_state.update(add_effects)

        # Check if goal facts are satisfied
        if goal_facts.issubset(current_state):
            return 1.0

        # Partial credit: fraction of goal facts achieved
        if self.allow_partial_credit and len(goal_facts) > 0:
            achieved = len(goal_facts.intersection(current_state))
            return achieved / len(goal_facts)

        return 0.0

    def verify(self, model_response: str, extra_env_info: dict) -> float:
        """Verifies step-by-step plan via PDDL simulation or string match fallback."""
        predicted_steps = self.extract_steps(model_response)
        if not predicted_steps:
            return 0.0

        meta = extra_env_info.get("verifier_meta", {})
        if isinstance(meta, dict) and "init" in meta and "goal" in meta and "actions" in meta:
            return self.simulate_pddl(predicted_steps, meta)

        # Fallback to ground truth string match
        expected_raw = extra_env_info.get("expected_answer", "")
        expected_steps = [s.strip() for s in str(expected_raw).split("\n") if s.strip()]
        if not expected_steps:
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
