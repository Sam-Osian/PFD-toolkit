# Gemma 4 Prompt Contracts

## 1. Runtime Expectations

Model runtime: local Ollama `gemma4` (exact tag to be pinned at deploy time).

Core rules:

1. Deterministic output configuration.
2. No chain-of-thought output.
3. Strict JSON output schema.

## 2. Issue Extraction Prompt

### System

You extract issue statements from coroners' report text.  
Return only JSON. Do not include reasoning or explanations.

### User Template

Extract distinct issue statements from the text below.

Rules:

1. Each issue must be one simple sentence.
2. Keep each sentence under 24 words.
3. Focus on systemic concern or failure, not narrative detail.
4. Avoid duplicates and near-duplicates.
5. Use neutral wording.
6. If no clear issue exists, return an empty list.

Output schema:

```json
{
  "issues": [
    "string"
  ]
}
```

Input text:

```
{{source_text}}
```

### Post-Parse Guards

1. Reject non-JSON or invalid schema.
2. Deduplicate case-insensitive.
3. Drop lines that are too long or too vague.
4. Cap issues per report (recommended initial cap: 8).

## 3. Cluster Label Prompt

### System

You name clusters of similar safety issues.  
Return only JSON. Do not include reasoning.

### User Template

Given the example issue statements below, propose:

1. a short cluster label (2–6 words)
2. a one-sentence description

Rules:

1. Label must be plain English.
2. Avoid legal or clinical jargon unless unavoidable.
3. Avoid "other", "misc", or generic filler labels.

Output schema:

```json
{
  "label": "string",
  "description": "string"
}
```

Examples:

```json
{{cluster_examples}}
```

## 4. Suggested Inference Settings (initial)

1. `temperature`: 0.0 to 0.2
2. `top_p`: 1.0
3. `max_tokens`: small bounded output
4. If supported: disable explicit reasoning trace output.

## 5. Prompt Versioning

Store prompt bundle hash as:

1. `extractor_version`
2. `labeler_version`

Do not overwrite old versions; append and roll forward.
