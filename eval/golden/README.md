# Golden Evaluation Dataset (`eval/golden/`)

## Schema (`GoldenItem`)

Each line in `golden.jsonl` is a JSON object conforming to `kb.domain.evaluation.GoldenItem`:
- `id`: Unique identifier matching `^q-\d{4}$`
- `question`: User query string
- `reference_answer`: Expected answer (must be empty `""` for `unanswerable` and `access_restricted`)
- `type`: `factual | multi_hop | table | unanswerable | access_restricted`
- `hub`: Target hub name (default `"default"`)
- `user_tags`: Simulated user ACL tags
- `source_uris`: Relative source URIs (resolved to document/chunk IDs at runtime)
- `evidence_spans`: Exact text expected to appear inside a single active chunk
- `restriction_kind`: `acl_tag | other_hub` (required for `access_restricted`)
- `control_for`: ID of the `access_restricted` item for which this item is the authorized control
- `provenance`: `manifest | llm | handwritten`
- `reviewed`: Boolean flag; **only `reviewed=true` items count toward CI evaluation gates**
- `reviewed_by` / `reviewed_at`: Reviewer identity and UTC timestamp

## Workflow

1. **Setup corpus** (idempotent):
   ```powershell
   uv run kb eval setup --plan eval/golden/corpus_plan.json --root data/raw/sample
   ```
2. **Build candidate golden items**:
   ```powershell
   uv run kb eval golden build --out eval/golden/golden.jsonl
   ```
3. **Preflight corpus check**:
   ```powershell
   uv run kb eval check-corpus --golden eval/golden/golden.jsonl --root data/raw/sample
   ```
4. **Review items**:
   - Interactive review: `uv run kb eval golden review --only-pending --reviewer "Your Name"`
   - Bulk-accept `manifest` items (never allowed for `llm` provenance):
     ```powershell
     uv run kb eval golden review --accept-provenance manifest --reviewer "Your Name" --yes
     ```
5. **Validate composition**:
   ```powershell
   uv run kb eval golden validate --golden eval/golden/golden.jsonl --min-items 64 --min-unanswerable-fraction 0.1
   ```
