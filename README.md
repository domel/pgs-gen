# PG-Schema Data Generator

Generate random graph data from PG-Schema files and export it to multiple formats:

- Standards-compliant GraphML with `pgs:list` metadata (default)
- Neo4j APOC-compatible GraphML with `attr.list`
- TinkerPop GraphML
- GraphSON 3.0 (no embedded types)
- Oracle Property Graph GraphSON

The generator supports PG-Schema label/property semantics (including `|`, `&`, `?`, `OPEN`, and `ABSTRACT`) and real list-valued properties via `LIST<datatype>`.

## Requirements

- Python 3.10+
- Optional: `faker` for realistic values

Install Faker (optional):

```bash
python3 -m pip install faker
```

## Usage

Basic (standards-compliant GraphML):

```bash
python3 pgs_generate.py examples/FraudGraphType.pgs 1 -o out.graphml
```

Neo4j APOC-compatible GraphML:

```bash
python3 pgs_generate.py examples/FraudGraphType.pgs 1 --format graphml-apoc -o out.graphml
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
  - `graphml` (default) -> standards-compliant GraphML with `pgs:list` metadata for list-valued properties
  - `graphml-apoc` / `apoc` -> Neo4j APOC-compatible GraphML with non-standard `attr.list`
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
    "format": "graphml"
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

## List Properties

`LIST<datatype>` is treated as a real property type and generates actual Python lists in memory and JSON arrays encoded into GraphML `<data>` text.

Default `graphml` mode stays standards-compliant:
- list-valued keys use `attr.type="string"`
- list values are serialized as JSON arrays inside the `<data>` element text
- list metadata is preserved on the key as `pgs:list="<inner-type>"`

`graphml-apoc` / `apoc` mode is intentionally non-standard and aimed at Neo4j/APOC interoperability:
- list-valued keys emit `attr.list="<type>"`
- this is practical for import workflows, but it is not formal GraphML compliance
- `DATE` and `DATETIME` lists are exported as string lists in GraphML-based formats

Only `LIST<datatype>` is treated as the explicit list feature. Bare `LIST` remains backward-compatible input, but it is not the structured list support described above.

Example schema with list-valued properties:

```bash
python3 pgs_generate.py examples/ListGraphType.pgs 1 -o out.graphml
python3 pgs_generate.py examples/ListGraphType.pgs 1 --format graphml-apoc -o out.apoc.graphml
```

## Tests

Run the test suite:

```bash
pytest -q
```

## Output Examples (abbreviated)

Standards-compliant GraphML (default):

```xml
<key id="aliases" for="node" attr.name="aliases" attr.type="string" pgs:list="string" />
<node id="n0">
  <data key="labels">:Person</data>
  <data key="aliases">["Alice","Al"]</data>
</node>
```

Neo4j APOC-compatible GraphML:

```xml
<key id="aliases" for="node" attr.name="aliases" attr.type="string" attr.list="string" />
<node id="n0" labels=":Person">
  <data key="labels">:Person</data>
  <data key="aliases">["Alice","Al"]</data>
</node>
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
