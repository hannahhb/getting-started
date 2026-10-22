"""Codabench entry point for the response-length method (placeholder).

The method (generate responses to the original problem, call it robust when
they are short) is being developed in ``research/``; until it is implemented
here, this predicts robust for every problem so the folder builds and runs.
"""


def are_robust(model_id: str, reasoning_effort: str, problems: list[str]) -> list[bool]:
    """Placeholder: predict robust for every problem, in input order."""
    return [True for _ in problems]
