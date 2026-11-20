---
name: acl-hub-filtering
description: How to write and test retrieval queries that enforce hub scope, org isolation, and document ACL tags inside SQL. Use whenever touching retrieval, repositories, search queries, document scope or hub membership, permissions, access control, or multi-hub behavior, even if the task does not mention security.
---

# Hub and ACL filtering

Invariant I1: access filtering happens inside the query. Never retrieve then filter.

## The rule set

A chunk is eligible only if all of these hold:

1. `chunks.is_active` and `documents.status = 'active'`
2. `documents.org_id = :org_id`
3. `documents.scope = 'global'` OR `(:hub_id, documents.id)` exists in `hub_documents`
4. `documents.acl_tags = '{}'` OR `documents.acl_tags && :user_tags`

`:org_id` and `:user_tags` come from the verified auth context. `:hub_id` comes from the route and must be authorized for the user. Never read these from request bodies or LLM output.

## Implementation guidance

- Put the filter in one function (`retrieval/filters.py`) that builds the SQL fragment or SQLAlchemy clause. Both the vector and full-text queries call it. No query touches `chunks` without it.
- Make the filter parameters a required typed object (`AccessContext`). A function that searches without one should not type-check.
- With HNSW indexes, a restrictive filter can return fewer than k rows. Verify with a test where few chunks pass the filter, and use iterative index scans or over-fetching so k eligible rows are returned when they exist.
- Keep the semantics fixed when optimizing. If you change the query shape, rerun the access tests.

## Required tests (negative first)

- User lacking a tag never sees a chunk from a tagged document, through either vector or full-text search.
- Hub A query never returns a hub-scoped document that belongs only to hub B.
- Global documents are visible from every hub in the org.
- Another org's documents are never visible.
- Inactive chunks and inactive documents are never returned.
- Changing a document from `hub` to `global` (and back) changes visibility without re-embedding.
- Fewer-than-k and filter-heavy cases still return k when k eligible rows exist.
