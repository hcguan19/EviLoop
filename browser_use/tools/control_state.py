"""Typed DOM-control inspection, deterministic mutation, and commit validation."""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, Field

from browser_use.browser import BrowserSession
from browser_use.dom.service import EnhancedDOMTreeNode

CONTROL_TAGS = {'input', 'select', 'textarea', 'button'}
CONTROL_ROLES = {'checkbox', 'combobox', 'listbox', 'option', 'radio', 'slider', 'spinbutton', 'textbox'}


class ControlOptionState(BaseModel):
	"""One visible option exposed by a select-like control."""

	value: str
	text: str
	selected: bool = False
	disabled: bool = False


class ControlState(BaseModel):
	"""A compact, non-secret snapshot of one live browser control."""

	index: int | None = None
	tag: str
	type: str = ''
	role: str = ''
	name: str = ''
	context: str = ''
	value: str | float | bool | list[str] | None = None
	checked: bool | None = None
	selected_values: list[str] = Field(default_factory=list)
	options: list[ControlOptionState] = Field(default_factory=list)
	minimum: float | None = None
	maximum: float | None = None
	step: float | None = None
	required: bool = False
	disabled: bool = False
	visible: bool = True
	valid: bool = True
	validation_message: str = ''
	contenteditable: bool = False


class CommitValidation(BaseModel):
	"""Program-level decision made before a submit-like click."""

	is_commit: bool = False
	allowed: bool = True
	reason: str = ''
	invalid_controls: list[dict[str, Any]] = Field(default_factory=list)


class MutationValidation(BaseModel):
	"""Program-level decision made before changing visible control evidence."""

	allowed: bool = True
	reason: str = ''


async def _resolve_object(browser_session: BrowserSession, node: EnhancedDOMTreeNode) -> tuple[Any, str, str]:
	cdp_session = await browser_session.cdp_client_for_node(node)
	result = await cdp_session.cdp_client.send.DOM.resolveNode(
		params={'backendNodeId': node.backend_node_id},
		session_id=cdp_session.session_id,
	)
	object_id = result.get('object', {}).get('objectId')
	if not object_id:
		raise ValueError('Could not resolve control node')
	return cdp_session.cdp_client, cdp_session.session_id, object_id


async def read_control_state(
	browser_session: BrowserSession,
	node: EnhancedDOMTreeNode,
	index: int | None = None,
) -> ControlState | None:
	"""Read the live state of a control without exposing password values."""

	client, session_id, object_id = await _resolve_object(browser_session, node)
	result = await client.send.Runtime.callFunctionOn(
		params={
			'objectId': object_id,
			'functionDeclaration': """
				function() {
					const el = this;
					const tag = (el.tagName || '').toLowerCase();
					const nativeType = (el.type || el.getAttribute?.('type') || '').toLowerCase();
					const jquerySlider = el.classList?.contains('ui-slider-handle')
						&& window.jQuery && window.jQuery(el).parent().hasClass('ui-slider');
					const type = jquerySlider ? 'range' : nativeType;
					const role = (el.getAttribute?.('role') || '').toLowerCase();
					const editable = el.isContentEditable === true;
					const password = tag === 'input' && type === 'password';
					const options = tag === 'select'
						? Array.from(el.selectedOptions || []).map(option => option.value || option.text)
						: [];
					const optionStates = tag === 'select'
						? Array.from(el.options || []).slice(0, 100).map(option => ({
							value: String(option.value || option.text || ''),
							text: String(option.text || '').trim().slice(0, 160),
							selected: Boolean(option.selected),
							disabled: Boolean(option.disabled)
						}))
						: [];
					const numeric = value => {
						if (value === null || value === undefined || value === '') return null;
						const parsed = Number(value);
						return Number.isFinite(parsed) ? parsed : null;
					};
					const style = window.getComputedStyle(el);
					const rect = el.getBoundingClientRect();
					return {
						tag,
						type,
						role,
						name: (() => {
							const labelledBy = el.getAttribute?.('aria-labelledby');
							const labelledText = labelledBy
								? labelledBy.split(/\\s+/)
									.map(id => document.getElementById(id)?.innerText || '')
									.join(' ').trim()
								: '';
							const explicitLabel = el.id
								? document.querySelector(`label[for="${CSS.escape(el.id)}"]`)?.innerText?.trim() || ''
								: '';
							const wrappingLabel = el.closest?.('label')?.innerText?.trim() || '';
							return el.getAttribute?.('aria-label') || labelledText || explicitLabel || wrappingLabel
								|| (tag === 'button' ? el.innerText?.trim() : '')
								|| el.getAttribute?.('placeholder') || el.name || el.id || el.innerText?.trim() || '';
						})(),
						context: (() => {
							const container = el.closest?.('label, tr, fieldset, [role="group"]') || el.parentElement;
							const text = String(container?.innerText || container?.textContent || '')
								.replace(/\\s+/g, ' ').trim();
							return text.slice(0, 300);
						})(),
						value: password ? '<redacted>' : jquerySlider
							? Number(window.jQuery(el).parent().slider('value')) : (
							type === 'checkbox' || type === 'radio'
								? Boolean(el.checked)
								: editable ? el.textContent : (el.value ?? el.getAttribute?.('aria-valuenow') ?? '')
						),
						checked: type === 'checkbox' || type === 'radio' ? Boolean(el.checked) : null,
						selected_values: options,
						options: optionStates,
						minimum: jquerySlider ? numeric(window.jQuery(el).parent().slider('option', 'min'))
							: numeric(el.min ?? el.getAttribute?.('aria-valuemin')),
						maximum: jquerySlider ? numeric(window.jQuery(el).parent().slider('option', 'max'))
							: numeric(el.max ?? el.getAttribute?.('aria-valuemax')),
						step: jquerySlider ? numeric(window.jQuery(el).parent().slider('option', 'step')) : numeric(el.step),
						required: Boolean(el.required || el.getAttribute?.('aria-required') === 'true'),
						disabled: Boolean(el.disabled || el.getAttribute?.('aria-disabled') === 'true'),
						visible: style.display !== 'none' && style.visibility !== 'hidden'
							&& rect.width > 0 && rect.height > 0,
						valid: typeof el.checkValidity === 'function' ? el.checkValidity() : true,
						validation_message: password ? '' : (el.validationMessage || ''),
						contenteditable: editable
					};
				}
			""",
			'returnByValue': True,
		},
		session_id=session_id,
	)
	value = result.get('result', {}).get('value')
	if not isinstance(value, dict):
		return None
	return ControlState(index=index, **value)


def is_control_node(node: EnhancedDOMTreeNode) -> bool:
	attrs = node.attributes or {}
	class_names = set(attrs.get('class', '').split())
	return (
		node.tag_name in CONTROL_TAGS
		or attrs.get('role', '').lower() in CONTROL_ROLES
		or 'ui-slider-handle' in class_names
		or attrs.get('contenteditable', '').lower() in {'', 'true', 'plaintext-only'}
		and 'contenteditable' in attrs
	)


def control_value_matches(state: ControlState, desired: str | bool | float | list[str]) -> bool:
	"""Return whether a live control snapshot matches an exact requested value."""

	if state.type in {'checkbox', 'radio'}:
		expected = desired if isinstance(desired, bool) else str(desired).lower() == 'true'
		return state.checked is expected
	if state.tag == 'select':
		if isinstance(desired, list):
			return sorted(map(str, desired)) == sorted(map(str, state.selected_values))
		return str(desired) == str(state.value) or str(desired) in state.selected_values
	if state.type in {'range', 'number'}:
		try:
			return abs(float(state.value) - float(desired)) < 1e-9
		except (TypeError, ValueError):
			return False
	return str(state.value) == str(desired)


async def inspect_controls(
	browser_session: BrowserSession,
	*,
	visible_only: bool = True,
	max_controls: int = 100,
) -> list[ControlState]:
	"""Return indexed live control states from the current selector map."""

	states: list[ControlState] = []
	selector_map = await browser_session.get_selector_map()
	for index, node in selector_map.items():
		if len(states) >= max_controls:
			break
		if not is_control_node(node):
			continue
		try:
			state = await read_control_state(browser_session, node, index)
		except Exception:
			continue
		if state is not None and (state.visible or not visible_only):
			states.append(state)
	return states


async def set_control_value(
	browser_session: BrowserSession,
	node: EnhancedDOMTreeNode,
	value: str | bool | float | list[str],
	index: int,
) -> tuple[ControlState | None, ControlState | None]:
	"""Set a control using native setters/events and return verified before/after states."""

	before = await read_control_state(browser_session, node, index)
	client, session_id, object_id = await _resolve_object(browser_session, node)
	await client.send.Runtime.callFunctionOn(
		params={
			'objectId': object_id,
			'functionDeclaration': """
				function(desired) {
					const el = this;
					const tag = (el.tagName || '').toLowerCase();
					const type = (el.type || '').toLowerCase();
					const jquerySlider = el.classList?.contains('ui-slider-handle')
						&& window.jQuery && window.jQuery(el).parent().hasClass('ui-slider');
					const emit = event => el.dispatchEvent(new Event(event, {bubbles: true}));
					el.focus();
					if (jquerySlider) {
						const numeric = Number(desired);
						if (!Number.isFinite(numeric)) throw new Error(`Expected numeric value, got: ${desired}`);
						window.jQuery(el).parent().slider('value', numeric);
					} else if (type === 'checkbox' || type === 'radio') {
						if (Array.isArray(desired)) throw new Error('Toggle controls require a boolean value.');
						const checked = desired === true || String(desired).toLowerCase() === 'true';
						if (el.checked !== checked) {
							if (type === 'radio' && !checked) {
								el.checked = false;
								emit('input');
								emit('change');
							} else {
								el.click();
							}
						}
					} else if (tag === 'select') {
						const wanted = (Array.isArray(desired) ? desired : [desired]).map(String);
						const matched = [];
						for (const option of Array.from(el.options)) {
							const selected = wanted.includes(option.value) || wanted.includes(option.text.trim());
							option.selected = selected;
							if (selected) matched.push(option.value || option.text.trim());
						}
						if (matched.length !== wanted.length) {
							throw new Error(`Expected ${wanted.length} option matches, found ${matched.length}.`);
						}
						emit('input');
						emit('change');
					} else if (el.isContentEditable) {
						if (Array.isArray(desired)) throw new Error('Editable controls require a scalar value.');
						el.focus();
						el.textContent = String(desired);
						emit('input');
						emit('change');
					} else {
						if (Array.isArray(desired)) throw new Error('Text and numeric controls require a scalar value.');
						let next = String(desired);
						if (type === 'range' || type === 'number') {
							let numeric = Number(desired);
							if (!Number.isFinite(numeric)) throw new Error(`Expected numeric value, got: ${desired}`);
							const min = el.min === '' ? -Infinity : Number(el.min);
							const max = el.max === '' ? Infinity : Number(el.max);
							numeric = Math.min(max, Math.max(min, numeric));
							next = String(numeric);
						}
						const prototype = tag === 'textarea'
							? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
						const descriptor = Object.getOwnPropertyDescriptor(prototype, 'value');
						if (descriptor?.set) descriptor.set.call(el, next);
						else el.value = next;
						emit('input');
						emit('change');
					}
					el.blur();
					return true;
				}
			""",
			'arguments': [{'value': value}],
			'returnByValue': True,
		},
		session_id=session_id,
	)
	after = await read_control_state(browser_session, node, index)
	return before, after


async def wait_for_control_state(
	browser_session: BrowserSession,
	node: EnhancedDOMTreeNode,
	index: int,
	*,
	seconds: float = 0.0,
	require_visible: bool = True,
	require_enabled: bool = True,
	stable_ms: int = 250,
) -> tuple[ControlState | None, ControlState | None]:
	"""Honor a minimum delay, then verify that a live control is ready and stable."""

	if seconds > 0:
		await asyncio.sleep(seconds)
	first = await read_control_state(browser_session, node, index)
	if first is None:
		return None, None
	if require_visible and not first.visible:
		return first, None
	if require_enabled and first.disabled:
		return first, None
	if stable_ms <= 0:
		return first, first
	await asyncio.sleep(stable_ms / 1000)
	second = await read_control_state(browser_session, node, index)
	if second is None:
		return first, None
	return first, second


async def validate_control_mutation(
	browser_session: BrowserSession,
	node: EnhancedDOMTreeNode,
) -> MutationValidation:
	"""Protect a non-empty visible copy source from accidental mutation."""

	client, session_id, object_id = await _resolve_object(browser_session, node)
	result = await client.send.Runtime.callFunctionOn(
		params={
			'objectId': object_id,
			'functionDeclaration': """
				function() {
					const el = this;
					const bodyText = String(document.body?.innerText || '').replace(/\\s+/g, ' ').toLowerCase();
					const exactTransferInstruction = /\\bcopy\\b.{0,160}\\b(paste|text\\s*box|field)\\b/.test(bodyText);
					const isVisibleSource = (el.tagName || '').toLowerCase() === 'textarea'
						&& String(el.value || '').length > 0;
					if (exactTransferInstruction && isVisibleSource) {
						return {
							allowed: false,
							reason: 'Visible copy source is immutable; read it and transfer its exact value to the target.'
						};
					}
					return {allowed: true};
				}
			""",
			'returnByValue': True,
		},
		session_id=session_id,
	)
	value = result.get('result', {}).get('value') or {}
	return MutationValidation(**value)


async def validate_commit_click(
	browser_session: BrowserSession,
	node: EnhancedDOMTreeNode,
) -> CommitValidation:
	"""Block commits that violate visible required-field or exact-transfer constraints."""

	client, session_id, object_id = await _resolve_object(browser_session, node)
	result = await client.send.Runtime.callFunctionOn(
		params={
			'objectId': object_id,
			'functionDeclaration': """
				function() {
					const el = this;
					const text = [
						el.innerText, el.value, el.name, el.id,
						el.getAttribute?.('aria-label'), el.getAttribute?.('title')
					].filter(Boolean).join(' ').toLowerCase();
					const commitWords = /\\b(submit|send|save|confirm|finish|complete|buy|order|checkout|sign)\\b/;
					const isCommit = el.type === 'submit' || commitWords.test(text);
					const isPurchaseCommit = /\\b(buy|order|checkout|purchase)\\b/.test(text);
					if (!isCommit) return {is_commit: false, allowed: true};
					const bodyText = String(document.body?.innerText || '').replace(/\\s+/g, ' ').toLowerCase();
					const exactTransferInstruction = /\\bcopy\\b.{0,160}\\b(paste|text\\s*box|field)\\b/.test(bodyText);
					if (exactTransferInstruction) {
						const visible = control => {
							const style = getComputedStyle(control);
							const rect = control.getBoundingClientRect();
							return !control.disabled && style.display !== 'none' && style.visibility !== 'hidden'
								&& rect.width > 0 && rect.height > 0;
						};
						const sources = Array.from(document.querySelectorAll('textarea')).filter(visible);
						const targets = Array.from(document.querySelectorAll(
							'input:not([type]), input[type="text"], input[type="search"], input[type="url"]'
						)).filter(visible);
						if (sources.length === 1 && targets.length === 1 && String(sources[0].value || '')) {
							if (String(sources[0].value) !== String(targets[0].value)) {
								const target = targets[0];
								const descriptor = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value');
								if (descriptor?.set) descriptor.set.call(target, String(sources[0].value));
								else target.value = String(sources[0].value);
								target.dispatchEvent(new Event('input', {bubbles: true}));
								target.dispatchEvent(new Event('change', {bubbles: true}));
								return {
									is_commit: true,
									allowed: false,
									reason: 'Visible exact-transfer instruction was auto-repaired; verify and retry commit.',
									invalid_controls: [{
										relation: 'exact_text_transfer',
										source: sources[0].getAttribute('aria-label') || sources[0].name || sources[0].id || '',
										target: targets[0].getAttribute('aria-label') || targets[0].name || targets[0].id || '',
										auto_repaired: true
									}]
								};
							}
						}
					}
					const form = el.form || el.closest?.('form');
					if (!form) return {is_commit: true, allowed: true, reason: 'No enclosing form.'};
					const invalid = Array.from(form.elements || [])
						.filter(control => {
							if (control.disabled || typeof control.checkValidity !== 'function' || control.checkValidity()) return false;
							const controlText = `${control.type || ''} ${control.name || ''} ${control.id || ''} ${control.placeholder || ''} ${control.getAttribute?.('aria-label') || ''}`.toLowerCase();
							const unrelatedSearchControl = isPurchaseCommit && (
								control.type === 'search' || /\\bsearch(?:[_ -]?query)?\\b/.test(controlText)
							);
							return !unrelatedSearchControl;
						})
						.map(control => ({
							tag: (control.tagName || '').toLowerCase(),
							type: (control.type || '').toLowerCase(),
							name: control.getAttribute?.('aria-label') || control.name || control.id || '',
							required: Boolean(control.required),
							message: control.type === 'password' ? 'Invalid required value.' : control.validationMessage
						}));
					return {
						is_commit: true,
						allowed: invalid.length === 0,
						reason: invalid.length ? 'Required form controls are invalid or incomplete.' : 'Form controls are valid.',
						invalid_controls: invalid
					};
				}
			""",
			'returnByValue': True,
		},
		session_id=session_id,
	)
	value = result.get('result', {}).get('value') or {}
	return CommitValidation(**value)
