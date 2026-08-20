# Field validation — excluded from this repository

The paper's external validation (§4.5–§4.7: the measured server-room /
building scenarios, Figures 10–11, the field rows of Figure 9, and the
field-validation tables) uses measured data from an operating AIDC campus,
provided by the data partner under an agreement that does not permit
redistribution. That part of the paper is therefore **outside the scope of
this open-source reproduction**: this repository contains only what is needed
to reproduce the public-data results.

Consequences you will see when running the pipeline:

- `figures/make_figs_cn.py` renders Figure 9 with the four public datasets
  only, and skips Figures 10 and 11 with an explanatory message.
- `analysis/clean_cqr_protocol_v2.py` runs its public Stage P and skips the
  field Stage F; `analysis/multiseed_rerun.py` runs public stages a/b and
  skips the field stage c; `analysis/aggregate_multiseed.py` aggregates the
  public sections only.
- The pre-test π audits cover the four public scenarios.

Researchers with authorized access to equivalent measured data can re-enable
these stages by placing derived 15-min CSVs under `data/field/`
(`field_room302_15min.csv`, `field_building_15min.csv` — a 15-min-resampled
IT-load series with a time index; see `common/config.py:FIELD_DATA_DIR`).
