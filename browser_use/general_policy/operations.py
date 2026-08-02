"""Validated, benchmark-independent operations compiled from task language."""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class VisibleTarget(BaseModel):
	"""A control grounded by visible semantics rather than a benchmark selector."""

	label: str = Field(min_length=1, max_length=200)
	roles: list[str] = Field(default_factory=list, max_length=10)


class NumericPredicate(BaseModel):
	"""A safe numeric condition evaluated against action-induced visible changes."""

	type: Literal['numeric'] = 'numeric'
	operator: Literal['modulo_equals', 'less_than', 'greater_than', 'equals']
	value: float
	divisor: int | None = Field(default=None, ge=1)

	def satisfied_by(self, observed: float) -> bool:
		"""Evaluate the bounded predicate without arbitrary code execution."""

		if self.operator == 'modulo_equals':
			return self.divisor is not None and observed % self.divisor == self.value
		if self.operator == 'less_than':
			return observed < self.value
		if self.operator == 'greater_than':
			return observed > self.value
		return observed == self.value


class RepeatUntilOperation(BaseModel):
	"""Repeat one reversible visible action until a visible predicate is satisfied."""

	type: Literal['repeat_until'] = 'repeat_until'
	action: VisibleTarget
	predicate: NumericPredicate
	max_attempts: int = Field(default=30, ge=1, le=50)
	poll_milliseconds: int = Field(default=40, ge=20, le=2000)


class CommitOperation(BaseModel):
	"""An irreversible visible action executed after all prior constraints pass."""

	type: Literal['commit'] = 'commit'
	target: VisibleTarget


Operation = Annotated[RepeatUntilOperation | CommitOperation, Field(discriminator='type')]


class TaskProgram(BaseModel):
	"""A small auditable program produced solely from user language."""

	operations: list[Operation] = Field(default_factory=list, max_length=20)
	parser: Literal['deterministic_language', 'structured_llm'] = 'deterministic_language'


_REPEAT_NUMERIC_PATTERN = re.compile(
	r'^(?:repeatedly\s+)?(?P<verb>generate|sample|roll|draw|refresh)\s+(?:a|an|the)?\s*'
	r'(?:(?P<parity>odd|even)\s+(?:number|value)|(?:number|value)\s+'
	r'(?P<comparison>less than|greater than|equal to)\s+(?P<threshold>-?\d+(?:\.\d+)?))'
	r',?\s+(?:then\s+)?(?:press|click|select)\s+(?P<commit>.+?)\.?$',
	flags=re.IGNORECASE,
)


def parse_task_program(instruction: str) -> TaskProgram:
	"""Compile supported language into generic operations without page or dataset identifiers."""

	match = _REPEAT_NUMERIC_PATTERN.match(re.sub(r'\s+', ' ', instruction).strip())
	if not match:
		return TaskProgram()
	parity = (match.group('parity') or '').casefold()
	comparison = (match.group('comparison') or '').casefold()
	if parity:
		predicate = NumericPredicate(
			operator='modulo_equals',
			divisor=2,
			value=0 if parity == 'even' else 1,
		)
	else:
		operator = {
			'less than': 'less_than',
			'greater than': 'greater_than',
			'equal to': 'equals',
		}[comparison]
		predicate = NumericPredicate(operator=operator, value=float(match.group('threshold')))
	return TaskProgram(
		operations=[
			RepeatUntilOperation(
				action=VisibleTarget(label=match.group('verb'), roles=['button']),
				predicate=predicate,
			),
			CommitOperation(target=VisibleTarget(label=match.group('commit'), roles=['button'])),
		]
	)
