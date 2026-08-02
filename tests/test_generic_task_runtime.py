from browser_use.tools.generic_task_runtime import (
	GenericEvidenceLedger,
	RecordSemanticDecision,
	StructuredVisibleRecord,
	aggregate_visible_dated_currency_records,
	build_generic_policy,
	completion_decision,
	extract_visible_total,
	extract_repeated_anchor_records,
	evidence_expansion_prompt,
	extract_latest_same_record_date,
	grounded_challenge_prompt,
	grounded_audit_prompt,
	GroundedRetrievalAudit,
	normalize_response_for_contract,
	preserve_visible_composite_values,
	rank_task_records,
	rank_expansion_records,
	rank_navigation_recovery_urls,
	rank_visible_comparison_candidates,
	resolve_comparison_spec,
	resolve_task_contract,
	semantic_decision_is_consistent,
	verify_retrieval_workflow,
)


def test_contract_resolution_is_dataset_independent() -> None:
	assert resolve_task_contract('Get the latest invoice total').operation == 'RETRIEVE'
	assert resolve_task_contract('Open the account settings page').operation == 'NAVIGATE'
	assert resolve_task_contract('Delete the draft message').operation == 'MUTATE'
	assert resolve_task_contract('Return how much I spent in January').operation == 'RETRIEVE'
	assert resolve_task_contract('Return the purchased item from order 123').operation == 'MUTATE'
	assert resolve_task_contract(
		'I recently moved, my address is 231 Willow Way, update my information accordingly'
	).operation == 'MUTATE'
	draft = resolve_task_contract('Fill out the contact form and leave it ready for review; do not submit it')
	assert draft.operation == 'MUTATE'
	assert draft.draft_only is True
	picture_frame = resolve_task_contract('Get the color of the picture frame I bought Sep 2022.')
	assert picture_frame.exhaustive is True
	assert picture_frame.requires_visual_evidence is False
	assert resolve_task_contract(
		'Read the serial number shown in the screenshot.'
	).requires_visual_evidence is True
	assert resolve_task_contract('Return how much I spent during Jan 2023').exhaustive is True
	assert resolve_task_contract('Return the date I last ordered olive bread').exhaustive is True
	assert resolve_task_contract(
		'Get how many orders I have. Return an object with keys "order_count" and "amount"'
	).output_shape == 'object'


def test_grounded_audit_accepts_object_shaped_retrieval() -> None:
	audit = GroundedRetrievalAudit.model_validate(
		{
			'retrieved_data': {'min': 1.46, 'max': 179.99},
			'complete': True,
			'confidence': 0.9,
			'reason': 'Visible product records cover the result set.',
		}
	)
	assert audit.retrieved_data == {'min': 1.46, 'max': 179.99}


def test_exhaustive_contract_and_policy() -> None:
	contract = resolve_task_contract('List all issue titles matching the label')
	assert contract.exhaustive is True
	policy = build_generic_policy(contract)
	assert 'pagination' in policy
	assert 'website' not in policy.casefold()
	mutation_policy = build_generic_policy(
		resolve_task_contract('Add 2 Hawaiian Bamboo Orchid Roots #zc50 to my wish list')
	)
	assert 'part of the identity' in mutation_policy
	assert 'prerequisite clauses' in mutation_policy
	assert 'largest visible page size' in mutation_policy


def test_comparison_spec_and_same_record_ranking() -> None:
	intent = (
		'Buy the highest rated product from the Ceiling light category within a budget above 1000. '
		'Discard any items in your cart if it is not empty.'
	)
	spec = resolve_comparison_spec(intent)
	assert spec.objective == 'highest_rating'
	assert spec.min_price == 1000
	assert spec.category_label == 'ceiling light'
	assert spec.required_terms == ['ceiling', 'light']
	records = [
		StructuredVisibleRecord(
			record_id='a',
			url='http://localhost/category',
			container_signature='card',
			text='36 Lights High Ceiling Chandelier Rating: 60% 2 Reviews $1,199.00 Add to Cart',
			controls=['Add to Cart'],
		),
		StructuredVisibleRecord(
			record_id='b',
			url='http://localhost/category',
			container_signature='card',
			text='40 X 138 High Ceiling Light Rating: 100% 4 Reviews $1,108.00 Add to Cart',
			controls=['Add to Cart'],
		),
		StructuredVisibleRecord(
			record_id='c',
			url='http://localhost/category',
			container_signature='card',
			text='Budget Ceiling Light Rating: 100% 20 Reviews $999.99 Add to Cart',
			controls=['Add to Cart'],
		),
	]
	ranked = rank_visible_comparison_candidates(intent, records)
	assert [item.record_text for item in ranked] == [
		'40 X 138 High Ceiling Light',
		'36 Lights High Ceiling Chandelier',
	]
	assert ranked[0].rating == 100
	assert ranked[0].price == 1108


def test_ledger_deduplicates_states_and_extracts_totals() -> None:
	ledger = GenericEvidenceLedger()
	text = 'Items 1 to 10 of 38\nNext'
	assert ledger.record(url='http://localhost/list?p=1', visible_text=text)
	assert not ledger.record(url='http://localhost/list?p=1', visible_text=text)
	assert ledger.snapshots[0].visible_total == 38
	assert ledger.snapshots[0].has_continuation is True
	assert extract_visible_total('Showing 1-20 of 75 results') == 75


def test_completion_gate_cannot_be_bypassed_by_repeating_done() -> None:
	contract = resolve_task_contract('List all matching records')
	ledger = GenericEvidenceLedger()
	ledger.record(url='http://localhost/list', visible_text='10 records\nNext')
	first = completion_decision(
		ledger,
		contract,
		final_text='{"status":"SUCCESS","retrieved_data":["a"]}',
		current_url='http://localhost/list',
		visible_text='10 records\nNext',
		start_url='http://localhost/list',
	)
	second = completion_decision(
		ledger,
		contract,
		final_text='{"status":"SUCCESS","retrieved_data":["a"]}',
		current_url='http://localhost/list',
		visible_text='10 records\nNext',
		start_url='http://localhost/list',
	)
	assert first is not None
	assert second is not None


def test_mutation_completion_requires_observed_request() -> None:
	contract = resolve_task_contract('Update my account information')
	ledger = GenericEvidenceLedger()
	ledger.record(url='http://localhost/account', visible_text='Account form')
	assert completion_decision(
		ledger,
		contract,
		intent='Update my account information',
		final_text='{"status":"SUCCESS"}',
		current_url='http://localhost/account',
		visible_text='Saved successfully',
		start_url='http://localhost/account',
		request_evidence=[],
	) is not None
	assert completion_decision(
		ledger,
		contract,
		intent='Update my account information',
		final_text='{"status":"SUCCESS"}',
		current_url='http://localhost/account',
		visible_text='Saved successfully',
		start_url='http://localhost/account',
		request_evidence=[{'method': 'POST', 'url': 'http://localhost/account'}],
	) is None


def test_mutation_completion_accepts_strong_persisted_ui_confirmation() -> None:
	contract = resolve_task_contract('Update my address to 231 Willow Way')
	ledger = GenericEvidenceLedger()
	ledger.record(url='http://localhost/account/edit', visible_text='Edit Address')
	ledger.record(
		url='http://localhost/account',
		visible_text='You saved the address. Default Billing Address 231 Willow Way',
	)
	assert completion_decision(
		ledger,
		contract,
		intent='Update my address to 231 Willow Way',
		final_text='{"status":"SUCCESS"}',
		current_url='http://localhost/account',
		visible_text='You saved the address. Default Billing Address 231 Willow Way',
		start_url='http://localhost',
		request_evidence=[],
	) is None


def test_purchase_completion_requires_purchase_confirmation() -> None:
	contract = resolve_task_contract('Buy the selected ceiling light')
	ledger = GenericEvidenceLedger()
	ledger.record(url='http://localhost/product', visible_text='Ceiling light')
	ledger.record(url='http://localhost/cart', visible_text='The item has been added to your shopping cart.')
	assert completion_decision(
		ledger,
		contract,
		intent='Buy the selected ceiling light',
		final_text='{"status":"SUCCESS"}',
		current_url='http://localhost/cart',
		visible_text='The item has been added to your shopping cart.',
		start_url='http://localhost',
		request_evidence=[],
	) is not None


def test_repeated_unconfirmed_purchase_requires_recovery_action() -> None:
	ledger = GenericEvidenceLedger()
	contract = resolve_task_contract('Buy the selected item')
	requests = [{'method': 'POST', 'url': 'https://shop.test/cart/add'}]
	first = completion_decision(
		ledger,
		contract,
		intent='Buy the selected item',
		final_text='{"status":"SUCCESS"}',
		current_url='https://shop.test/checkout',
		visible_text='Shipping Methods: Sorry, no quotes are available for this order at this time',
		start_url='https://shop.test',
		request_evidence=requests,
	)
	second = completion_decision(
		ledger,
		contract,
		intent='Buy the selected item',
		final_text='{"status":"SUCCESS"}',
		current_url='https://shop.test/checkout',
		visible_text='Shipping Methods: Sorry, no quotes are available for this order at this time',
		start_url='https://shop.test',
		request_evidence=requests,
	)

	assert first is not None
	assert second is not None
	assert 'Do not call done again' in second
	assert 'different recovery action' in second


def test_navigation_completion_requires_destination_constraints() -> None:
	contract = resolve_task_contract('Open women shoes filtered to under 25')
	ledger = GenericEvidenceLedger()
	assert completion_decision(
		ledger,
		contract,
		intent='Open women shoes filtered to under 25',
		final_text='{"status":"SUCCESS"}',
		current_url='http://localhost/category/women/shoes?price=0-25',
		visible_text='Women Shoes Price 0-25',
		start_url='http://localhost/',
	) is None
	assert completion_decision(
		ledger,
		contract,
		intent='Open women shoes filtered to under 25',
		final_text='{"status":"SUCCESS"}',
		current_url='http://localhost/category/shoes',
		visible_text='Women Shoes',
		start_url='http://localhost/',
	) is not None


def test_navigation_upper_bound_rejects_narrower_subset_and_wrong_hierarchy() -> None:
	contract = resolve_task_contract('Open women shoes filtered to under 25')
	ledger = GenericEvidenceLedger()
	assert completion_decision(
		ledger,
		contract,
		intent='Open women shoes filtered to under 25',
		final_text='{"status":"SUCCESS"}',
		current_url='http://localhost/clothing/women/shoes?price=10-20',
		visible_text='Women Shoes Price 10-20 under 25',
		start_url='http://localhost/',
	) is not None
	assert completion_decision(
		ledger,
		contract,
		intent='Open women shoes filtered to under 25',
		final_text='{"status":"SUCCESS"}',
		current_url='http://localhost/clothing?price=0-25',
		visible_text='Women Shoes Price 0-25',
		start_url='http://localhost/',
	) is not None
	assert completion_decision(
		ledger,
		contract,
		intent='Open women shoes filtered to under 25',
		final_text='{"status":"SUCCESS"}',
		current_url='http://localhost/clothing/women/shoes?price=0-25',
		visible_text='Women Shoes Price 0-25',
		start_url='http://localhost/',
	) is None


def test_navigation_recovery_prefers_full_hierarchy_and_preserves_exact_range() -> None:
	records = [
		StructuredVisibleRecord(
			record_id='nav',
			url='http://localhost/',
			container_signature='nav',
			text='Clothing Women Shoes',
			links=[
				{'label': 'Shoes', 'href': '/shoes'},
				{'label': 'Women Shoes', 'href': '/clothing/women/shoes.html'},
			],
		)
	]
	ranked = rank_navigation_recovery_urls(
		'Open women shoes filtered to under 25',
		records,
		current_url='http://localhost/search?price=10-20',
	)
	assert ranked[0] == 'http://localhost/clothing/women/shoes.html?price=0-25'
	assert rank_navigation_recovery_urls(
		'Open women shoes filtered to under 25',
		[],
		current_url='http://localhost/clothing-shoes-jewelry.html?cat=144&price=20-29',
	)[0] == 'http://localhost/clothing-shoes-jewelry/women/shoes.html?price=0-25'
	assert rank_navigation_recovery_urls(
		'Open women shoes filtered to under 25',
		[],
		current_url='http://localhost/clothing-shoes-jewelry/women.html?cat=144&price=20-29',
	)[0] == 'http://localhost/clothing-shoes-jewelry/women/shoes.html?price=0-25'


def test_historical_date_requires_exact_entity_and_same_record() -> None:
	records = [
		StructuredVisibleRecord(
			record_id='wrong-substring',
			url='http://localhost/order/2',
			container_signature='body',
			text='Order Date: January 20, 2023 Gingerbread cookies',
		),
		StructuredVisibleRecord(
			record_id='matching',
			url='http://localhost/order/1',
			container_signature='body',
			text='Order Date: December 12, 2022 Items: Whole Foods Market Bread Batard Olive, tea',
		),
		StructuredVisibleRecord(
			record_id='whole-page-false-binding',
			url='http://localhost/orders',
			container_signature='visible-page-body',
			text='Order Date: May 17, 2023 ' + ('unrelated field ' * 200) + 'Popular search terms: olive bread',
		),
	]
	intent = 'Return the date I last ordered my olive bread.'
	assert extract_latest_same_record_date(intent, records) == '2022-12-12'
	ledger = GenericEvidenceLedger(records=records)
	verdict = verify_retrieval_workflow(intent, ['2023-01-20'], ledger)
	assert verdict.accepted is False
	assert verdict.failure_kind == 'record_binding'
	assert verdict.verified_values == ['2022-12-12']


def test_history_expansion_ranks_transaction_details_above_unrelated_products() -> None:
	records = [
		StructuredVisibleRecord(
			record_id='product',
			url='http://localhost/',
			container_signature='product-card',
			text='Fresh bread selection',
			links=[{'label': 'Open product', 'href': '/product/44'}],
		),
		StructuredVisibleRecord(
			record_id='order',
			url='http://localhost/order-history',
			container_signature='order-row',
			text='Order 1001 12/12/22 Complete',
			links=[{'label': 'View Order', 'href': '/order/1001'}],
		),
	]
	ranked = rank_expansion_records('Return the date I last ordered olive bread.', records)
	assert ranked[0].record_id == 'order'


def test_ledger_retains_large_cross_page_record_set() -> None:
	ledger = GenericEvidenceLedger()
	for index in range(1500):
		ledger.record_structured(
			url='http://localhost/list',
			records=[{'container_signature': 'tr', 'text': f'Order {index} Complete total {index}.00'}],
		)
	assert len(ledger.records) == 1500
	assert ledger.records[0].text.startswith('Order 0 ')


def test_visible_dated_currency_aggregate_uses_exact_window_and_deduplicates() -> None:
	intent = (
		'Today is June 12, 2023. Get how many complete orders I have over the past year, and the total amount. '
		'Return an object with keys "order_count" and "amount" only.'
	)
	records = [
		StructuredVisibleRecord(
			record_id='a',
			url='http://localhost/orders',
			container_signature='tr',
			text='000001 6/12/22 $10.25 Complete View Order',
		),
		StructuredVisibleRecord(
			record_id='a-duplicate',
			url='http://localhost/orders?p=2',
			container_signature='tr',
			text='000001 6/12/22 $10.25 Complete View Order',
		),
		StructuredVisibleRecord(
			record_id='b',
			url='http://localhost/orders',
			container_signature='tr',
			text='000002 6/11/22 $20.00 Complete View Order',
		),
		StructuredVisibleRecord(
			record_id='c',
			url='http://localhost/orders',
			container_signature='tr',
			text='000003 5/1/23 $3.50 Pending View Order',
		),
		StructuredVisibleRecord(
			record_id='d',
			url='http://localhost/orders',
			container_signature='tr',
			text='000004 6/12/23 $7.75 Complete View Order',
		),
	]
	assert aggregate_visible_dated_currency_records(intent, records) == {
		'order_count': 2,
		'amount': 18.0,
	}


def test_response_normalization_does_not_invent_answers() -> None:
	contract = resolve_task_contract('Get the invoice number')
	assert normalize_response_for_contract({'status': 'SUCCESS', 'retrieved_data': 'A-1'}, contract) == {
		'task_type': 'RETRIEVE',
		'status': 'SUCCESS',
		'retrieved_data': ['A-1'],
		'error_details': None,
	}
	empty = normalize_response_for_contract({'status': 'SUCCESS', 'retrieved_data': None}, contract)
	assert empty['status'] == 'UNKNOWN_ERROR'


def test_grounded_audit_prompt_has_only_generic_record_rules() -> None:
	system, user = grounded_audit_prompt('Find matching values', ['draft'], 'Row A\nValue 3')
	assert 'same record' in system
	assert 'filter qualifying records first' in system
	assert 'shopping' not in system.casefold()
	assert 'review' not in system.casefold()
	assert 'Row A' in user


def test_challenge_prompt_checks_omissions_and_wrong_record_binding() -> None:
	audit = GroundedRetrievalAudit(
		retrieved_data=['A'],
		complete=True,
		confidence=0.8,
		reason='first pass',
		evidence=['row A'],
	)
	system, user = grounded_challenge_prompt('Find all matching values', audit, 'row A\nrow B')
	assert 'omitted records' in system
	assert 'same record' in system
	assert 'site rule' in system
	assert 'row B' in user


def test_repeated_anchor_scanner_discovers_records_without_field_names() -> None:
	text = """
Title one
Body alpha
Owner by Alice
Title two
Body beta
Owner by Bob
Title three
Body gamma
Owner by Carol
"""
	records = extract_repeated_anchor_records(text)
	assert any('Body alpha' in record and 'Owner by Alice' in record for record in records)
	assert any('Body beta' in record and 'Owner by Bob' in record for record in records)
	assert any('Body gamma' in record and 'Owner by Carol' in record for record in records)


def test_structured_records_are_deduplicated_and_ranked_into_evidence() -> None:
	ledger = GenericEvidenceLedger()
	records = [
		{'container_signature': 'li|listitem|entry', 'text': 'Alpha invoice total 12', 'controls': ['Open']},
		{'container_signature': 'li|listitem|entry', 'text': 'Beta invoice total 20', 'controls': ['Open']},
	]
	assert ledger.record_structured(url='http://localhost/list', records=records) == 2
	assert ledger.record_structured(url='http://localhost/list', records=records) == 0
	evidence = ledger.compact_evidence('Get the beta invoice total')
	assert 'BROWSER-DERIVED STRUCTURED RECORDS' in evidence
	assert 'Beta invoice total 20' in evidence
	assert 'CONTROLS: ["Open"]' in evidence


def test_record_recall_uses_predicate_terms_and_rejects_self_contradiction() -> None:
	records = [
		StructuredVisibleRecord(
			record_id='matching',
			url='http://localhost',
			container_signature='article||',
			text='Only half the ear fits inside the cups. Review by Alice',
		),
		StructuredVisibleRecord(
			record_id='template-only',
			url='http://localhost',
			container_signature='article||',
			text='The printed instructions use small type. Review by Bob',
		),
	]
	ranked = rank_task_records(
		'Get names of reviewers who mention ear cups being small for the product on the current page',
		records,
	)
	assert [record.record_id for record in ranked] == ['matching']
	assert not semantic_decision_is_consistent(
		RecordSemanticDecision(
			relevant=True,
			projected_values=['Bob'],
			confidence=0.95,
			reason='This does not mention ear cups and therefore does not satisfy the request.',
			evidence='The printed instructions use small type.',
		)
	)
	intent = 'Get names of reviewers who mention ear cups being small'
	assert semantic_decision_is_consistent(
		RecordSemanticDecision(
			relevant=True,
			projected_values=['Alice'],
			confidence=0.95,
			reason='Only half of the ear fits inside the cups.',
			evidence='These are not over-the-ear cups; only half my ear fits inside.',
		),
		intent,
	)
	explicit_intent = "Get reviewers who mention 'print quality' explicitly"
	assert not semantic_decision_is_consistent(
		RecordSemanticDecision(
			relevant=True,
			projected_values=['Bob'],
			confidence=1.0,
			reason='The reviewer reports poor printing.',
			evidence='Poorly designed printer with lousy output. Review by Bob.',
		),
		explicit_intent,
	)
	unquoted_intent = 'Get reviewers who mention print quality explicitly with a rating of 3 or less stars'
	assert semantic_decision_is_consistent(
		RecordSemanticDecision(
			relevant=True,
			projected_values=['Alice'],
			confidence=1.0,
			reason='Both predicates are visible.',
			evidence='The print quality was average. Rating 20%. Review by Alice.',
		),
		unquoted_intent,
		record_text='The print quality was average. Rating 20%. Review by Alice.',
	)
	assert not semantic_decision_is_consistent(
		RecordSemanticDecision(
			relevant=True,
			projected_values=['Bob'],
			confidence=1.0,
			reason='Printing was poor.',
			evidence='Color printing was poor. Rating 20%. Review by Bob.',
		),
		unquoted_intent,
		record_text='Color printing was poor. Rating 20%. Review by Bob.',
	)
	assert semantic_decision_is_consistent(
		RecordSemanticDecision(
			relevant=True,
			projected_values=['Alice'],
			confidence=1.0,
			reason='The literal phrase is visible.',
			evidence='The print quality was average. Review by Alice.',
		),
		explicit_intent,
	)
	assert not semantic_decision_is_consistent(
		RecordSemanticDecision(
			relevant=True,
			projected_values=['Bob'],
			confidence=0.95,
			reason='The record contains both concepts.',
			evidence=('over ear headphones ' + ('unrelated text ' * 30) + 'small printed instructions'),
		),
		intent,
	)


def test_evidence_expansion_uses_only_visible_record_links() -> None:
	record = StructuredVisibleRecord(
		record_id='order-row',
		url='http://localhost/orders',
		container_signature='tr||',
		text='1001 1/29/23 Complete',
		links=[{'label': 'View', 'href': 'http://localhost/orders/1001'}],
	)
	system, user = evidence_expansion_prompt('Get the item color bought on Jan 29', [record])
	assert 'hidden data' in system
	assert 'order-row' in user
	assert 'http://localhost/orders/1001' in user


def test_expansion_ranking_normalizes_visible_numeric_dates() -> None:
	records = [
		StructuredVisibleRecord(
			record_id='wrong-month',
			url='http://localhost/orders',
			container_signature='tr||',
			text='1001 5/2/23 Complete',
			links=[{'label': 'View', 'href': 'http://localhost/orders/1001'}],
		),
		StructuredVisibleRecord(
			record_id='matching-date',
			url='http://localhost/orders',
			container_signature='tr||',
			text='1002 1/29/23 Complete',
			links=[{'label': 'View Details', 'href': 'http://localhost/orders/1002'}],
		),
	]
	ranked = rank_expansion_records('How much did I spend during January 29, 2023?', records)
	assert ranked[0].record_id == 'matching-date'


def test_visible_composite_value_is_not_silently_trimmed() -> None:
	evidence = 'Product\nColor\nMist 16*24\nSKU B09'
	assert preserve_visible_composite_values(
		['Mist'],
		['Mist 16*24'],
		evidence,
		'Get the color of the frame',
	) == ['Mist 16*24']
	assert preserve_visible_composite_values(
		['Mist'],
		[],
		evidence,
		'Get the color of the frame',
	) == ['Mist 16*24']
	assert preserve_visible_composite_values(
		['10'],
		['10 USD'],
		'Total 10 USD',
		'Return the value as a number only',
	) == ['10']
