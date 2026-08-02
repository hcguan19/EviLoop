"""Benchmark-independent candidate grounding over observable evidence."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from browser_use.tools.task_policy import (
	RankedCandidate,
	TaskObservation,
	VisibleCandidate,
	extract_task_requirements,
	rank_visible_candidates,
)


class GroundingExample(BaseModel):
	"""One offline decision point expressed through the shared observation contract."""

	example_id: str
	objective: str
	observation: TaskObservation
	positive_identifiers: set[str] = Field(min_length=1)
	operation: str = ''


class GroundingPrediction(BaseModel):
	"""Auditable ranking and retrieval metrics for one decision point."""

	example_id: str
	predicted_identifier: str | None
	positive_identifiers: set[str]
	ranked_identifiers: list[str]
	positive_rank: int | None
	top1_correct: bool
	recall_at_5: bool
	recall_at_10: bool
	reciprocal_rank: float
	candidate_count: int
	operation: str = ''
	focused_objective: str = ''


def focus_current_instruction(objective: str) -> str:
	"""Prefer the latest user turn when the observation contains a dialogue transcript."""

	user_turns = re.findall(r'(?im)^user:\s*(.+)$', objective)
	if user_turns:
		return re.sub(r'\s+', ' ', user_turns[-1]).strip()
	return re.sub(r'\s+', ' ', objective).strip()


def _expected_interaction(objective: str, operation: str) -> str:
	operation = operation.casefold()
	if operation in {'type', 'text_input', 'paste'}:
		return 'text_input'
	if operation in {'select', 'change'}:
		return 'select'
	if operation == 'click':
		return 'click'
	normalized = objective.casefold()
	if re.search(r'\b(?:enter|type|paste|fill|add|write)\b', normalized):
		return 'text_input'
	if re.search(r'\b(?:choose|select|pick)\b', normalized):
		return 'select'
	return 'click' if re.search(r'\b(?:click|press|open)\b', normalized) else ''


def _interaction_adjustment(candidate: VisibleCandidate, expected: str) -> float:
	"""Apply generic role priors without relying on page selectors or benchmark identity."""

	tag = str(candidate.metadata.get('tag') or '').casefold()
	attributes = str(candidate.metadata.get('attributes') or '').casefold()
	interactive = any(
		marker in attributes
		for marker in ('contenteditable', 'is_clickable', 'onclick', 'tabindex', 'role=button', 'role="button')
	)
	if expected == 'text_input':
		if tag in {'input', 'textarea'} or 'contenteditable' in attributes:
			return 4.0
		if tag in {'div', 'span'} and interactive:
			return 2.5
	if expected == 'select':
		if tag in {'select', 'option'} or 'combobox' in attributes:
			return 3.0
	if expected == 'click':
		if tag in {'button', 'a', 'input', 'select', 'option', 'svg'} or interactive:
			return 1.5
	if tag in {'html', 'body', 'main', 'header', 'footer'}:
		return -1.25
	if tag in {'div', 'section', 'article'} and not interactive:
		return -0.45
	return 0.0


def rank_grounding_example(example: GroundingExample) -> tuple[GroundingPrediction, list[RankedCandidate]]:
	"""Rank recorded visible candidates with the same policy used for live pages."""

	focused_objective = focus_current_instruction(example.objective)
	requirements = extract_task_requirements(focused_objective)
	ranked = rank_visible_candidates(requirements, example.observation.candidates)
	expected = _expected_interaction(focused_objective, example.operation)
	for item in ranked:
		item.score = round(item.score + _interaction_adjustment(item.candidate, expected), 4)
	ranked.sort(key=lambda item: (item.eligible, item.score), reverse=True)
	return prediction_from_ranked(example, focused_objective, ranked), ranked


def prediction_from_ranked(
	example: GroundingExample, focused_objective: str, ranked: list[RankedCandidate]
) -> GroundingPrediction:
	"""Score any policy-produced ordering against references kept outside the policy."""

	ranked_identifiers = [item.candidate.identifier for item in ranked]
	positive_rank = next(
		(index for index, identifier in enumerate(ranked_identifiers, start=1) if identifier in example.positive_identifiers),
		None,
	)
	return GroundingPrediction(
		example_id=example.example_id,
		predicted_identifier=ranked_identifiers[0] if ranked_identifiers else None,
		positive_identifiers=example.positive_identifiers,
		ranked_identifiers=ranked_identifiers,
		positive_rank=positive_rank,
		top1_correct=positive_rank == 1,
		recall_at_5=positive_rank is not None and positive_rank <= 5,
		recall_at_10=positive_rank is not None and positive_rank <= 10,
		reciprocal_rank=0.0 if positive_rank is None else 1.0 / positive_rank,
		candidate_count=len(ranked_identifiers),
		operation=example.operation,
		focused_objective=focused_objective,
	)


def summarize_grounding_predictions(predictions: list[GroundingPrediction]) -> dict[str, float | int]:
	"""Aggregate standard retrieval metrics without benchmark-specific scoring logic."""

	count = len(predictions)
	if not count:
		return {'examples': 0, 'top1_accuracy': 0.0, 'recall_at_5': 0.0, 'recall_at_10': 0.0, 'mrr': 0.0}
	return {
		'examples': count,
		'top1_accuracy': sum(item.top1_correct for item in predictions) / count,
		'recall_at_5': sum(item.recall_at_5 for item in predictions) / count,
		'recall_at_10': sum(item.recall_at_10 for item in predictions) / count,
		'mrr': sum(item.reciprocal_rank for item in predictions) / count,
	}
