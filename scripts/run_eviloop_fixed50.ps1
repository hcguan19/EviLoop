param(
	[Parameter(Mandatory = $true)]
	[ValidateSet('miniwob', 'webshop')]
	[string]$Benchmark,
	[string]$Python = 'python',
	[string]$BaseUrl = $env:QWEN_CHAT_BASE_URL,
	[string]$ApiKey = $env:QWEN_CHAT_API_KEY,
	[string]$Model = $env:QWEN_CHAT_MODEL,
	[int]$Seed = 17,
	[int]$MaxParallel = 1,
	[string]$ResultDirectory = ''
)

$ErrorActionPreference = 'Stop'
$repository = Split-Path -Parent $PSScriptRoot
$taskDirectory = Join-Path $repository "evaluation\tasks\${Benchmark}50_seed17"

if (-not (Test-Path -LiteralPath $taskDirectory)) {
	throw "Task directory not found: $taskDirectory"
}
if (-not $BaseUrl -or -not $ApiKey -or -not $Model) {
	throw 'Set QWEN_CHAT_BASE_URL, QWEN_CHAT_API_KEY, and QWEN_CHAT_MODEL or pass the matching parameters.'
}
if (-not $ResultDirectory) {
	$ResultDirectory = Join-Path $repository "benchmark_results\eviloop_${Benchmark}50_seed${Seed}"
}

New-Item -ItemType Directory -Path $ResultDirectory -Force | Out-Null
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONPATH = $repository
$env:QWEN_CHAT_BASE_URL = $BaseUrl
$env:QWEN_CHAT_API_KEY = $ApiKey
$env:QWEN_CHAT_MODEL = $Model
$env:EVAL_METHOD_VERSION = 'eviloop'
$env:EVAL_MAX_PARALLEL = [string]$MaxParallel
$env:EVAL_RESULTS_JSONL = Join-Path $ResultDirectory 'results.jsonl'
$env:EVAL_RESUME = 'true'
$env:EVAL_SEED = [string]$Seed
$env:EVAL_EXPERIMENT_LABEL = "eviloop_${Benchmark}50_seed${Seed}"
$env:EVAL_USE_VISION = 'false'
$env:EVAL_VISUAL_CONTEXT_MODE = 'html_first'
$env:EVAL_SHORT_UI_STRATEGY = 'true'
$env:EVAL_MAX_HISTORY_ITEMS = '6'
$env:EVAL_MAX_CLICKABLE_ELEMENTS_LENGTH = '7000'
$env:EVAL_MAX_ACTIONS_PER_STEP = '1'
$env:EVAL_NAVIGATION_TIMEOUT_SECONDS = '30'
$env:QWEN_MAX_COMPLETION_TOKENS = '640'

if ($Benchmark -eq 'webshop') {
	$env:EVAL_WEBSHOP_WEB_ONLY = 'true'
	$env:EVAL_GOAL_AWARE_TASK_STRATEGY = 'true'
	$env:EVAL_WEBSHOP_MAX_STEPS = '12'
	$env:EVAL_AGENT_RUN_TIMEOUT_SECONDS = '600'
}
else {
	Remove-Item Env:EVAL_WEBSHOP_WEB_ONLY -ErrorAction SilentlyContinue
	Remove-Item Env:EVAL_GOAL_AWARE_TASK_STRATEGY -ErrorAction SilentlyContinue
	$env:EVAL_AGENT_RUN_TIMEOUT_SECONDS = '300'
}

Remove-Item Env:EVAL_COMPARISON_METHOD -ErrorAction SilentlyContinue
Remove-Item Env:BROWSER_USE_API_KEY -ErrorAction SilentlyContinue
Remove-Item Env:GOOGLE_API_KEY -ErrorAction SilentlyContinue

& $Python (Join-Path $repository 'tests\ci\evaluate_tasks.py') $taskDirectory *>&1 |
	Tee-Object -FilePath (Join-Path $ResultDirectory 'runner.log')

exit $LASTEXITCODE
