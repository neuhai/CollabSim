Answer the interviewer probe below. Respond with JSON only.
{construct_line}
Question: {prompt}

Observation:
{observation_json}

Return JSON with the following shape:
{{
  "answer": "...",
  "confidence": 0.0,
  "structured_fields": {{}}
}}
