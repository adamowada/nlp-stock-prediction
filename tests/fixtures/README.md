# Fixtures

Fixtures support deterministic tests for the prediction research assistant.

Fixture data should model source payloads, normalized evidence, tool artifacts, SQLite rows, and
report outputs without requiring live network access.

Preferred layout:

```text
tests/fixtures/
  raw/{provider}/{scenario}.json
  normalized/evidence/{scenario}.json
  tools/{tool_name}/{scenario}.json
  sqlite/{scenario}/
  prediction_candidates/{scenario}.json
  reports/{scenario}/expected_report.json
  reports/{scenario}/expected_report.md
```

Raw provider fixtures should include:

- fixture version;
- provider name;
- scenario;
- recorded timestamp;
- request or query;
- response path;
- provider metadata;
- redaction metadata.

Normalized fixtures should use the same public contracts as production code. Scenarios must not mix
dates or universes unless the test explicitly exercises stale or contradictory evidence.
