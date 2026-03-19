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

Explicit type references with `@`:

```bash
python3 pgs_generate.py schema.pgs 1 --type-ref-prefix @ -o out.graphml
```

## CLI Options

- `--format`: output format
  - `apoc` (default) -> Neo4j APOC GraphML
  - `graphml-tp` -> TinkerPop GraphML
  - `graphson3` -> GraphSON 3.0 without embedded types
  - `oracle-graphson` -> Oracle Property Graph GraphSON
- `-f/--fake`: use Faker for more realistic values
- `--seed`: deterministic random data
- `--type-ref-prefix PREFIX`: optional explicit type-reference prefix in label expressions, for example `@PersonType`
- `--open-extra`: when `OPEN` is used, force extra labels/properties
- `--open-extra-count N`: how many extra labels/properties to add per `OPEN` element
- `--mutation-config FILE`: JSON file with non-conforming mutation probabilities, optionally per type
- `--mutation-report FILE`: JSON report with mutation summary and per-object mutation details
- `--mutation-fresh-node-prob P`: probability of adding a fresh unrelated node after generating a node
- `--mutation-fresh-edge-prob P`: probability of adding a fresh unrelated edge after generating an edge
- `--mutation-extra-label-prob P`: probability of adding a fresh label to a closed node/edge
- `--mutation-extra-prop-prob P`: probability of adding a fresh property to a closed node/edge record
- `--mutation-bad-optional-prop-prob P`: probability of writing an invalid value for an `OPTIONAL` property
- `--mutation-wrong-prop-datatype-prob P`: probability of writing an invalid value for any schema-defined property
- `--mutation-missing-required-prop-prob P`: probability of omitting a required property
- `--mutation-missing-required-label-prob P`: probability of omitting a required label
- `--mutation-typo-label-prob P`: probability of introducing a typo into a required label
- `--mutation-typo-prop-key-prob P`: probability of introducing a typo into a schema property key
- `--tp-labels-prop`: for `graphml-tp`, include full label set as a `labels` property if multiple labels exist
- `--graphson-wrap`: for `graphson3`, output a single JSON object with `{"vertices":[...]}` instead of line-delimited

## Non-Conforming Data

All mutation probabilities default to `0.0`, so the generator remains conforming unless mutations are enabled.

Example with direct CLI flags:

```bash
python3 pgs_generate.py examples/FraudGraphType.pgs 1 \
  --mutation-extra-label-prob 0.2 \
  --mutation-extra-prop-prob 0.2 \
  --mutation-bad-optional-prop-prob 0.1 \
  --mutation-wrong-prop-datatype-prob 0.1 \
  --mutation-typo-label-prob 0.05 \
  --mutation-typo-prop-key-prob 0.05
```

Write a separate mutation report:

```bash
python3 pgs_generate.py examples/FraudGraphType.pgs 1 \
  --mutation-extra-label-prob 0.2 \
  --mutation-report mutation-report.json
```

Example `mutations.json`:

```json
{
  "defaults": {
    "extra_label_probability": 0.1,
    "extra_property_probability": 0.1,
    "fresh_node_probability": 0.05,
    "typo_label_probability": 0.02
  },
  "types": {
    "Person": {
      "missing_required_property_probability": 0.25,
      "wrong_property_datatype_probability": 0.2
    },
    "Knows": {
      "invalid_optional_property_probability": 0.5,
      "missing_required_label_probability": 0.5,
      "typo_property_key_probability": 0.2
    }
  }
}
```

Use it like this:

```bash
python3 pgs_generate.py examples/FraudGraphType.pgs 1 --mutation-config mutations.json
```

The mutation report is stored outside the generated graph and contains:
- `meta`: schema path, graph type, seed, scale, format
- `summary`: totals, mutated node/edge counts, mutation counts by kind
- `objects`: per-node/per-edge mutation details with `before`/`after` values when available

Example `mutation-report.json`:

```json
{
  "meta": {
    "schema": "examples/FraudGraphType.pgs",
    "graph_type": "FraudGraphType",
    "seed": 42,
    "scale": 1,
    "format": "apoc"
  },
  "summary": {
    "nodes_total": 50,
    "edges_total": 60,
    "mutated_nodes": 8,
    "mutated_edges": 11,
    "by_kind": {
      "extra_label": 3,
      "fresh_node": 2,
      "typo_property_key": 2,
      "wrong_property_datatype": 4
    }
  },
  "objects": [
    {
      "id": "n12",
      "kind": "node",
      "type": "Person",
      "mutations": [
        {
          "kind": "extra_label",
          "before": null,
          "after": "MutatedLabel12345"
        },
        {
          "kind": "wrong_property_datatype",
          "property": "age",
          "expected": "INT",
          "before": 35,
          "after": "invalid_age_12"
        }
      ]
    },
    {
      "id": "e7",
      "kind": "edge",
      "type": "Knows",
      "source": "n2",
      "target": "n9",
      "mutations": [
        {
          "kind": "typo_property_key",
          "property": "since",
          "before": "since",
          "after": "sicne",
          "value": "2018-01-01"
        }
      ]
    }
  ]
}
```

Only mutated objects are included in `objects`. The `summary.by_kind` counters track applied mutations, not just the number of affected objects.

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
