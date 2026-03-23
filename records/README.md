# Records

Each folder is a saved experiment result.

## Naming Convention

```
YYYYMMDD_<type>_<eval>_<desc>
```

**type**
- `snn` — trained SNN evaluated with LinearSVC
- `raw-baseline` — raw spike data directly into LinearSVC (no SNN)
- `rand-baseline` — random-weight SNN into LinearSVC (untrained)

**eval** — how features are extracted before LinearSVC
- `tsum` — `sum(dim=0).flatten()`: sum over time, keep full spatial → ~70K features
- `gmean` — `sum(dim=0).mean(dim=(-1,-2))`: sum over time, mean over spatial → (C,) = 32 features
- `smean` — `mean(dim=(-1,-2))`: mean over spatial only, keep time → (T×C) = 1,920 features

**desc** — brief note, e.g. `101cls`, `2cls`, `100cls`
