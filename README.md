# PG-Schema Data Generator

Generate random graph data from PG-Schema files and export it to multiple formats:

- Neo4j APOC GraphML (default)
- TinkerPop GraphML
- GraphSON 3.0 (no embedded types)
- Oracle Property Graph GraphSON

The generator supports PG-Schema label/property semantics (including `|`, `&`, `?`, `OPEN`, and `ABSTRACT`).

## Requirements

- Python 3.10+
- Optional: `faker` for realistic values

Install Faker (optional):

```bash
python3 -m pip install faker
```

## Usage

Basic (Neo4j APOC GraphML):

```bash
python3 pgs_generate.py examples/FraudGraphType.pgs 1 -o out.graphml
```

TinkerPop GraphML:

```bash
python3 pgs_generate.py examples/FraudGraphType.pgs 1 --format graphml-tp -o out.graphml
```

GraphSON 3.0 (line-delimited, no embedded types):

```bash
python3 pgs_generate.py examples/FraudGraphType.pgs 1 --format graphson3 -o out.graphson
```

Oracle GraphSON:

```bash
python3 pgs_generate.py examples/FraudGraphType.pgs 1 --format oracle-graphson -o out.oracle.graphson
```

## CLI Options

- `--format`: output format
  - `apoc` (default) -> Neo4j APOC GraphML
  - `graphml-tp` -> TinkerPop GraphML
  - `graphson3` -> GraphSON 3.0 without embedded types
  - `oracle-graphson` -> Oracle Property Graph GraphSON
- `-f/--fake`: use Faker for more realistic values
- `--seed`: deterministic random data
- `--open-extra`: when `OPEN` is used, force extra labels/properties
- `--open-extra-count N`: how many extra labels/properties to add per `OPEN` element
- `--tp-labels-prop`: for `graphml-tp`, include full label set as a `labels` property if multiple labels exist
- `--graphson-wrap`: for `graphson3`, output a single JSON object with `{"vertices":[...]}` instead of line-delimited

## Notes on Semantics

- `&` requires all labels.
- `|` is an exclusive choice between label branches.
- `?` makes a label optional.
- `OPEN` allows extra labels/properties; use `--open-extra` to force them.
- `ABSTRACT` types are never instantiated directly.

## Tests

Run the test suite:

```bash
pytest -q
```

## Output Examples (abbreviated)

Neo4j APOC GraphML:

```xml
<node id="n0" labels=":Person">
  <data key="labels">:Person</data>
  <data key="name">Alice</data>
</node>
<edge id="e0" source="n0" target="n1" label="KNOWS">
  <data key="label">KNOWS</data>
</edge>
```

TinkerPop GraphML:

```xml
<node id="n0">
  <data key="labelV">Person</data>
  <data key="name">Alice</data>
</node>
<edge id="e0" source="n0" target="n1">
  <data key="labelE">KNOWS</data>
</edge>
```

GraphSON 3.0 (line-delimited, no embedded types):

```json
{"id":"n0","label":"Person","properties":{"name":[{"id":0,"value":"Alice"}]}}
```

Oracle GraphSON:

```json
{"graph":{"mode":"NORMAL","vertices":[{"_id":"n0","_type":"vertex","name":"Alice"}],"edges":[{"_id":"e0","_type":"edge","_outV":"n0","_inV":"n1","_label":"KNOWS"}]}}
```
