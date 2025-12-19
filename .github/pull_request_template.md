## What

- 

## Why

- 

## How to verify

- [ ] `python -m pytest tests/ -v`
- [ ] Notebook still runs top→bottom (or explain why not)

## Stability invariants (do not regress)

- [ ] No silent ignoring of public API params
- [ ] `LAND_COVER_CLASSES` remains the single source of truth
- [ ] Paths work from notebooks/CLI/API (no CWD dependence)
