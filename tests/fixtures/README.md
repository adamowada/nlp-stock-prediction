# Fixture Contract

Phase 0 freezes fixture shape, not fixture content. Later contract and e2e tests should use this layout:

```text
tests/fixtures/
  raw/{provider}/{scenario}.json
  normalized/provider_results/{provider}/{scenario}.json
  normalized/evidence/{scenario}.json
  extraction/{scenario}.json
  analysis/{scenario}.json
  scoring/{scenario}.json
  reports/{scenario}/expected_report.json
  reports/{scenario}/expected_report.md
```

Raw provider fixtures should include `fixture_version`, `provider_name`, `scenario`,
`recorded_at`, `request`, `response_path`, `provider_metadata`, and `redactions`.
Normalized fixtures should use the same public Pydantic contracts as production code.
