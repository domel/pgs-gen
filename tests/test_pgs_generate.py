import datetime as dt
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pgs_generate


def _load_schema(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _is_value_of_type(value, prop_type):
    t = prop_type.upper()
    if t in {"INT", "INT32", "INT64", "LONG"}:
        if isinstance(value, bool):
            return False
        if isinstance(value, int):
            return True
        try:
            int(value)
            return True
        except (ValueError, TypeError):
            return False
    if t in {"DOUBLE", "FLOAT", "DECIMAL"}:
        if isinstance(value, bool):
            return False
        if isinstance(value, (int, float)):
            return True
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False
    if t in {"BOOLEAN", "BOOL"}:
        if isinstance(value, bool):
            return True
        return value in {"true", "false"}
    if t == "DATE":
        try:
            dt.date.fromisoformat(value)
            return True
        except ValueError:
            return False
    if t in {"DATETIME", "TIMESTAMP"}:
        try:
            dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
            return True
        except ValueError:
            return False
    if t == "STRING":
        return isinstance(value, str)
    return isinstance(value, str)


def _assert_edges_respect_endpoint_labels(schema_path, seed):
    text = _load_schema(schema_path)
    node_types, edge_types, graph_types = pgs_generate.parse_schema(text)
    graph_type = graph_types[0]
    rng = pgs_generate.random.Random(seed)

    nodes, edges, _, _ = pgs_generate.generate_instances(graph_type, node_types, edge_types, 1, rng)
    semantics = pgs_generate.SchemaSemantics(node_types, edge_types)

    nodes_by_id = {n.node_id: n for n in nodes}

    for edge in edges:
        source = nodes_by_id[edge.source]
        target = nodes_by_id[edge.target]
        options = semantics.eval_edge_type(edge.type_name)
        assert options
        assert any(
            pgs_generate._labels_conform(set(edge.labels), opt.edge)
            and pgs_generate._record_conforms(edge.props, opt.edge.record_spec)
            and pgs_generate._labels_conform(set(source.labels), opt.source)
            and pgs_generate._record_conforms(source.props, opt.source.record_spec)
            and pgs_generate._labels_conform(set(target.labels), opt.target)
            and pgs_generate._record_conforms(target.props, opt.target.record_spec)
            for opt in options
        )


def _assert_property_types(schema_path, seed):
    text = _load_schema(schema_path)
    node_types, edge_types, graph_types = pgs_generate.parse_schema(text)
    graph_type = graph_types[0]
    rng = pgs_generate.random.Random(seed)

    nodes, edges, _, _ = pgs_generate.generate_instances(graph_type, node_types, edge_types, 1, rng)
    semantics = pgs_generate.SchemaSemantics(node_types, edge_types)

    for node in nodes:
        options = semantics.eval_node_type(node.type_name)
        assert options
        assert any(
            pgs_generate._labels_conform(set(node.labels), opt)
            and pgs_generate._record_conforms(node.props, opt.record_spec)
            for opt in options
        )

    for edge in edges:
        options = semantics.eval_edge_type(edge.type_name)
        assert options
        assert any(
            pgs_generate._labels_conform(set(edge.labels), opt.edge)
            and pgs_generate._record_conforms(edge.props, opt.edge.record_spec)
            for opt in options
        )


def test_parse_catalog_schema():
    text = _load_schema("examples/CatalogGraphType.pgs")
    node_types, edge_types, graph_types = pgs_generate.parse_schema(text)

    assert "CatalogType" in node_types
    assert "HasResourceType" in edge_types
    assert len(graph_types) == 1
    assert graph_types[0].name == "CatalogGraphType"

    element_names = {e.name for e in graph_types[0].elements if hasattr(e, "name")}
    assert "CatalogType" in element_names
    assert "HasResourceType" in element_names


def test_generate_catalog_graphml_counts_and_labels():
    text = _load_schema("examples/CatalogGraphType.pgs")
    node_types, edge_types, graph_types = pgs_generate.parse_schema(text)
    graph_type = graph_types[0]
    rng = pgs_generate.random.Random(42)

    nodes, edges, node_prop_names, edge_prop_names = pgs_generate.generate_instances(
        graph_type, node_types, edge_types, 1, rng
    )

    assert len(nodes) == 60
    assert len(edges) == 60

    catalog_nodes = [n for n in nodes if "Catalog" in n.labels]
    assert catalog_nodes
    for node in catalog_nodes:
        assert "CatalogType" not in node.labels

    tree = pgs_generate.build_graphml(nodes, edges, node_prop_names, edge_prop_names)
    root = tree.getroot()
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
    labels_keys = root.findall("g:key[@for='node'][@attr.name='labels']", ns)
    assert labels_keys

    graph = root.find("g:graph", ns)
    assert graph is not None
    sample_node = graph.find("g:node", ns)
    assert sample_node is not None
    assert "labels" in sample_node.attrib


def test_generate_fraud_graph_labels():
    text = _load_schema("examples/FraudGraphType.pgs")
    node_types, edge_types, graph_types = pgs_generate.parse_schema(text)
    graph_type = graph_types[0]
    rng = pgs_generate.random.Random(7)

    nodes, edges, _, _ = pgs_generate.generate_instances(graph_type, node_types, edge_types, 1, rng)

    assert len(nodes) == 50
    assert len(edges) == 60

    allowed = {"owns", "uses", "charge", "deposit", "withdraw"}
    assert all(edge.labels[0] in allowed for edge in edges)

    customer_nodes = [n for n in nodes if {"Person", "Customer"} <= set(n.labels)]
    assert customer_nodes
    for node in customer_nodes:
        assert "name" in node.props
        assert "c_id" in node.props


def test_edges_respect_endpoint_labels():
    _assert_edges_respect_endpoint_labels("examples/CatalogGraphType.pgs", 3)
    _assert_edges_respect_endpoint_labels("examples/FraudGraphType.pgs", 5)
    _assert_edges_respect_endpoint_labels("examples/star-wars_pgschema.pgs", 7)


def test_property_types_match_schema():
    _assert_property_types("examples/CatalogGraphType.pgs", 11)
    _assert_property_types("examples/FraudGraphType.pgs", 13)
    _assert_property_types("examples/star-wars_pgschema.pgs", 17)


def test_fake_value_formats():
    faker_module = pytest.importorskip("faker")
    faker = faker_module.Faker()
    faker.seed_instance(42)
    rng = pgs_generate.random.Random(42)

    date_val = pgs_generate.generate_value("DATE", rng, 1, 0, "created", faker)
    assert dt.date.fromisoformat(date_val)

    dt_val = pgs_generate.generate_value("DATETIME", rng, 1, 0, "edited", faker)
    dt.datetime.strptime(dt_val, "%Y-%m-%dT%H:%M:%S.%fZ")

    int_val = pgs_generate.generate_value("INT", rng, 1, 0, "id", faker)
    assert int(int_val) >= 0

    str_val = pgs_generate.generate_value("STRING", rng, 1, 0, "name", faker)
    assert isinstance(str_val, str)
    assert str_val

    gender_val = pgs_generate.generate_value("STRING", rng, 1, 0, "gender", faker)
    assert gender_val in {"male", "female", "non-binary", "other"}

    age_val = pgs_generate.generate_value("INT", rng, 1, 0, "age", faker)
    assert 0 <= int(age_val) <= 100

    birth_val = pgs_generate.generate_value("DATE", rng, 1, 0, "birthdate", faker)
    assert dt.date.fromisoformat(birth_val)


def test_star_wars_schema_parsing_and_generation():
    text = _load_schema("examples/star-wars_pgschema.pgs")
    node_types, edge_types, graph_types = pgs_generate.parse_schema(text)

    assert {"Character", "Film", "Planet", "Species", "Starship", "Vehicle"} <= set(node_types.keys())
    assert {"APPEARED_IN", "HOMEWORLD", "OF", "PILOT"} <= set(edge_types.keys())
    assert len(graph_types) == 1

    graph_type = graph_types[0]
    rng = pgs_generate.random.Random(21)
    nodes, edges, _, _ = pgs_generate.generate_instances(graph_type, node_types, edge_types, 1, rng)

    assert nodes
    assert edges
    allowed = {"APPEARED_IN", "HOMEWORLD", "OF", "PILOT"}
    assert all(edge.labels[0] in allowed for edge in edges)


def test_edge_and_labels_graphml():
    schema = """
    CREATE GRAPH TYPE G LOOSE {
      (N: A & B),
      (:A & B)-[:REL_A & REL_B]->(:A & B)
    };
    """
    node_types, edge_types, graph_types = pgs_generate.parse_schema(schema)
    graph_type = graph_types[0]
    rng = pgs_generate.random.Random(9)
    nodes, edges, node_prop_names, edge_prop_names = pgs_generate.generate_instances(
        graph_type, node_types, edge_types, 1, rng
    )
    assert nodes
    assert edges
    assert all({"REL_A", "REL_B"} <= set(edge.labels) for edge in edges)

    tree = pgs_generate.build_graphml(nodes, edges, node_prop_names, edge_prop_names)
    root = tree.getroot()
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
    labels_key = root.find("g:key[@for='edge'][@attr.name='labels']", ns)
    assert labels_key is not None
    graph = root.find("g:graph", ns)
    edge_el = graph.find("g:edge", ns)
    assert edge_el is not None
    labels_data = edge_el.find("g:data[@key='e_labels']", ns)
    if labels_data is None:
        labels_data = edge_el.find("g:data[@key='labels']", ns)
    assert labels_data is not None
    assert labels_data.text and ":REL_A" in labels_data.text and ":REL_B" in labels_data.text


def test_cli_open_extra_generates_graphml(tmp_path):
    schema = """
    CREATE GRAPH TYPE G LOOSE {
      (N: A OPEN {id INT, OPEN}),
      (:A OPEN)-[:REL OPEN {OPEN}]->(:A OPEN)
    };
    """
    schema_path = tmp_path / "schema.pgs"
    schema_path.write_text(schema, encoding="utf-8")
    out_path = tmp_path / "out.graphml"
    result = subprocess.run(
        [
            sys.executable,
            "pgs_generate.py",
            str(schema_path),
            "1",
            "--open-extra",
            "--open-extra-count",
            "2",
            "-o",
            str(out_path),
        ],
        cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        check=True,
        capture_output=True,
        text=True,
    )
    assert out_path.exists()
    tree = ET.parse(out_path)
    root = tree.getroot()
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
    graph = root.find("g:graph", ns)
    assert graph is not None
    assert graph.find("g:node", ns) is not None

    keys = {}
    for key_el in root.findall("g:key", ns):
        keys[key_el.attrib["id"]] = key_el.attrib.get("attr.name", "")

    nodes = graph.findall("g:node", ns)
    edges = graph.findall("g:edge", ns)
    assert nodes
    assert edges

    for node in nodes:
        labels = (node.attrib.get("labels") or "").split(":")
        labels = [label for label in labels if label]
        extra_labels = [label for label in labels if label.startswith("ExtraLabel")]
        assert len(extra_labels) == 2

        extra_props = []
        for data in node.findall("g:data", ns):
            attr_name = keys.get(data.attrib.get("key", ""), "")
            if attr_name.startswith("extra_prop_"):
                extra_props.append(attr_name)
        assert len(extra_props) == 2

    edge_label_data = None
    for edge in edges:
        for data in edge.findall("g:data", ns):
            attr_name = keys.get(data.attrib.get("key", ""), "")
            if attr_name == "labels":
                edge_label_data = data.text or ""
                break
        assert edge_label_data is not None
        labels = edge_label_data.split(":")
        labels = [label for label in labels if label]
        extra_labels = [label for label in labels if label.startswith("ExtraLabel")]
        assert len(extra_labels) == 2

        extra_props = []
        for data in edge.findall("g:data", ns):
            attr_name = keys.get(data.attrib.get("key", ""), "")
            if attr_name.startswith("extra_prop_"):
                extra_props.append(attr_name)
        assert len(extra_props) == 2


def test_open_extras_flag_adds_labels_and_props():
    schema = """
    CREATE GRAPH TYPE G LOOSE {
      (N: A OPEN {id INT, OPEN}),
      (:A OPEN)-[:REL OPEN {OPEN}]->(:A OPEN)
    };
    """
    node_types, edge_types, graph_types = pgs_generate.parse_schema(schema)
    graph_type = graph_types[0]
    rng = pgs_generate.random.Random(11)
    nodes, edges, _, _ = pgs_generate.generate_instances(
        graph_type, node_types, edge_types, 1, rng, force_open_extras=1
    )

    assert any(label.startswith("ExtraLabel") for node in nodes for label in node.labels)
    assert any(key.startswith("extra_prop_") for node in nodes for key in node.props.keys())
    assert any(label.startswith("ExtraLabel") for edge in edges for label in edge.labels)
    assert any(key.startswith("extra_prop_") for edge in edges for key in edge.props.keys())


def test_open_extra_count_adds_multiple_extras():
    schema = """
    CREATE GRAPH TYPE G LOOSE {
      (N: A OPEN {id INT, OPEN}),
      (:A OPEN)-[:REL OPEN {OPEN}]->(:A OPEN)
    };
    """
    node_types, edge_types, graph_types = pgs_generate.parse_schema(schema)
    graph_type = graph_types[0]
    rng = pgs_generate.random.Random(12)
    nodes, edges, _, _ = pgs_generate.generate_instances(
        graph_type, node_types, edge_types, 1, rng, force_open_extras=2
    )
    for node in nodes:
        extra_labels = [label for label in node.labels if label.startswith("ExtraLabel")]
        assert len(extra_labels) == 2
        extra_props = [key for key in node.props if key.startswith("extra_prop_")]
        assert len(extra_props) == 2

    for edge in edges:
        extra_labels = [label for label in edge.labels if label.startswith("ExtraLabel")]
        assert len(extra_labels) == 2
        extra_props = [key for key in edge.props if key.startswith("extra_prop_")]
        assert len(extra_props) == 2


def test_graphml_tinkerpop_format():
    schema = """
    CREATE GRAPH TYPE G LOOSE {
      (N: Person {age INT}),
      (:Person)-[:KNOWS {since DATE}]->(:Person)
    };
    """
    node_types, edge_types, graph_types = pgs_generate.parse_schema(schema)
    graph_type = graph_types[0]
    rng = pgs_generate.random.Random(14)
    nodes, edges, _, _ = pgs_generate.generate_instances(graph_type, node_types, edge_types, 1, rng)

    tree = pgs_generate.build_graphml_tinkerpop(nodes, edges)
    root = tree.getroot()
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}

    labelv = root.find("g:key[@id='labelV']", ns)
    labele = root.find("g:key[@id='labelE']", ns)
    assert labelv is not None
    assert labele is not None
    assert labelv.attrib.get("attr.type") == "string"
    assert labele.attrib.get("attr.type") == "string"

    age_key = root.find("g:key[@id='age']", ns)
    assert age_key is not None
    assert age_key.attrib.get("attr.type") == "int"

    graph = root.find("g:graph", ns)
    node = graph.find("g:node", ns)
    data_label = node.find("g:data[@key='labelV']", ns)
    assert data_label is not None
    assert data_label.text


def test_graphml_tinkerpop_labels_property():
    schema = """
    CREATE GRAPH TYPE G LOOSE {
      (N: A & B),
      (:A & B)-[:REL]->(:A & B)
    };
    """
    node_types, edge_types, graph_types = pgs_generate.parse_schema(schema)
    graph_type = graph_types[0]
    rng = pgs_generate.random.Random(16)
    nodes, edges, _, _ = pgs_generate.generate_instances(graph_type, node_types, edge_types, 1, rng)

    tree = pgs_generate.build_graphml_tinkerpop(nodes, edges, include_labels_prop=True)
    root = tree.getroot()
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
    labels_key = root.find("g:key[@attr.name='labels']", ns)
    assert labels_key is not None

    graph = root.find("g:graph", ns)
    node = graph.find("g:node", ns)
    labels_data = node.find("g:data[@key='labels']", ns)
    assert labels_data is not None
    assert "A" in (labels_data.text or "") and "B" in (labels_data.text or "")


def test_graphson3_no_embedded_types(tmp_path):
    schema = """
    CREATE GRAPH TYPE G LOOSE {
      (N: Person {age INT}),
      (:Person)-[:KNOWS {since DATE}]->(:Person)
    };
    """
    node_types, edge_types, graph_types = pgs_generate.parse_schema(schema)
    graph_type = graph_types[0]
    rng = pgs_generate.random.Random(15)
    nodes, edges, _, _ = pgs_generate.generate_instances(graph_type, node_types, edge_types, 1, rng)

    out_path = tmp_path / "graph.graphson"
    pgs_generate.write_graphson3_no_types(nodes, edges, out_path)

    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines
    first = json.loads(lines[0])
    assert "id" in first and "label" in first
    assert "@type" not in lines[0]
    if "properties" in first:
        prop_entry = next(iter(first["properties"].values()))
        assert isinstance(prop_entry, list)
        assert "value" in prop_entry[0]


def test_graphson3_wrap(tmp_path):
    schema = """
    CREATE GRAPH TYPE G LOOSE {
      (N: Person {age INT})
    };
    """
    node_types, edge_types, graph_types = pgs_generate.parse_schema(schema)
    graph_type = graph_types[0]
    rng = pgs_generate.random.Random(18)
    nodes, edges, _, _ = pgs_generate.generate_instances(graph_type, node_types, edge_types, 1, rng)
    out_path = tmp_path / "graph.graphson"
    pgs_generate.write_graphson3_no_types(nodes, edges, out_path, wrap=True)

    content = out_path.read_text(encoding="utf-8").strip()
    payload = json.loads(content)
    assert "vertices" in payload
    assert isinstance(payload["vertices"], list)


def test_oracle_graphson_format(tmp_path):
    schema = """
    CREATE GRAPH TYPE G LOOSE {
      (N: Person {age INT}),
      (:Person)-[:KNOWS {since DATE}]->(:Person)
    };
    """
    node_types, edge_types, graph_types = pgs_generate.parse_schema(schema)
    graph_type = graph_types[0]
    rng = pgs_generate.random.Random(19)
    nodes, edges, _, _ = pgs_generate.generate_instances(graph_type, node_types, edge_types, 1, rng)
    out_path = tmp_path / "oracle.graphson"
    pgs_generate.write_oracle_graphson(nodes, edges, out_path)

    content = out_path.read_text(encoding="utf-8").strip()
    payload = json.loads(content)
    assert "graph" in payload
    graph = payload["graph"]
    assert graph.get("mode") == "NORMAL"
    assert isinstance(graph.get("vertices"), list)
    assert isinstance(graph.get("edges"), list)

    vertex = graph["vertices"][0]
    assert "_id" in vertex
    assert vertex.get("_type") == "vertex"

    edge = graph["edges"][0]
    assert edge.get("_type") == "edge"
    assert "_outV" in edge and "_inV" in edge and "_label" in edge
