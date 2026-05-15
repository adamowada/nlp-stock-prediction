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
  tools/universe_discovery/{scenario}.json
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

## Universe Discovery Fixtures

Instrument-Universe Stage universe fixtures are small contract-shaped JSON files under
`tests/fixtures/tools/universe_discovery/`.

Use them for deterministic tests of:

- broad asset-class representation;
- provider ID and namespace preservation;
- tradability/access evidence provenance;
- explicit ambiguous, unsupported, and unavailable resolutions;
- watchlist-driven universe requests.

Universe fixture dates should be fixed, usually `2026-05-13`, and source URLs may use
`https://example.test/` or raw fixture identifiers. These fixtures are not live provider captures and
must not imply that an instrument is tradable today.
