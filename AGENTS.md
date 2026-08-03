# Repository notebook ownership

- Treat every checked-in notebook as user-authored source of truth, including
  prompts, settings, markdown, cell order and IDs, outputs, and experiment cells.
- Never regenerate or replace an existing tracked notebook as a side effect of
  an unrelated implementation change.
- Never enable a builder's force-overwrite escape hatch for a tracked notebook.
  Builders may emit reference notebooks only to new temporary or explicitly
  requested paths.
- Apply notebook changes as narrow, marker-based migrations. Name the cells or
  markers being changed, preserve every other cell byte-for-byte where practical,
  and inspect the notebook diff before committing.
- If a broad regeneration is genuinely required, stop and obtain the user's
  explicit approval after explaining exactly which authored notebook state would
  be replaced.
