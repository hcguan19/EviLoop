"""Orchestration facade for the shared capability policy."""

from __future__ import annotations

from pydantic import BaseModel

from browser_use.general_policy.operations import TaskProgram, parse_task_program
from browser_use.tools.task_policy import PolicyRoute, TaskObservation, TaskRequirements, extract_task_requirements, route_observation


class PolicyContext(BaseModel):
	"""One normalized policy input independent of the source benchmark."""

	requirements: TaskRequirements
	observation: TaskObservation
	route: PolicyRoute
	program: TaskProgram


def build_policy_context(task: str, observation: TaskObservation) -> PolicyContext:
	"""Build the same policy context for browser, visual, desktop, and replay adapters."""

	return PolicyContext(
		requirements=extract_task_requirements(task),
		observation=observation,
		route=route_observation(observation),
		program=parse_task_program(task),
	)
