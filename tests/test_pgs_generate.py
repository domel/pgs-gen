import datetime as dt
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pgs_generate


MUTATION_SCHEMA = """
CREATE GRAPH TYPE G STRICT {
  (Person: Person {id INT, OPTIONAL nick STRING}),
  (:Person)-[Knows: KNOWS {since DATE, OPTIONAL weight INT}]->(:Person)
};
"""

TYPO_SCHEMA = """
CREATE GRAPH TYPE G STRICT {
  (Person: Person {id INT}),
  (:Person)-[Knows: KNOWS {since DATE}]->(:Person)
};
"""

KEYWORD_PROPERTY_SCHEMA = """
CREATE NODE TYPE
( EntityType : Entity {
  OPTIONAL type String
} );

CREATE GRAPH TYPE ICIJ LOOSE {
  EntityType
};
"""

TYPE_REF_SCHEMA = """
CREATE GRAPH TYPE G STRICT {
  (PostType: Post {id STRING}),
  (CommentType: Comment {id STRING}),
  (PersonType: Person {id STRING}),
  (: @PostType | @CommentType)-[HasCreatorType: HAS_CREATOR]->(: @PersonType),
  (: @CommentType)-[ReplyOfType: REPLY_OF]->(: @CommentType | @PostType)
};
"""

APOC_TYPED_GRAPHML_SCHEMA = """
CREATE NODE TYPE
(testNodeType : LabelOne {
  propkeyInteger INT,
  propKeyLong LONG,
  propKeyDouble DOUBLE,
  propKeyBoolean BOOLEAN,
  propKeyList LIST<STRING>
});

CREATE GRAPH TYPE testGraphType STRICT {
    testNodeType
}
"""

LIST_PROPERTY_SCHEMA = """
CREATE NODE TYPE
(ListNodeType : SomeLabel {
  propKeyStrings LIST<STRING>,
  propKeyInts LIST<LONG>,
  propKeyFloats LIST<DOUBLE>,
  propKeyBooleans LIST<BOOLEAN>,
  propKeyDates LIST<DATE>,
  propKeyDatetimes LIST<TIMESTAMP>
});

CREATE GRAPH TYPE testGraphType STRICT {
    ListNodeType
}
"""


def _load_schema(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _is_value_of_type(value, prop_type):
    inner_type = pgs_generate._list_inner_prop_type(prop_type)
    if inner_type is not None:
        return isinstance(value, list) and all(_is_value_of_type(item, inner_type) for item in value)
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


def test_keyword_property_name_parses_and_generates():
    node_types, edge_types, graph_types = pgs_generate.parse_schema(KEYWORD_PROPERTY_SCHEMA)

    entity_type = node_types["EntityType"]
    assert len(entity_type.spec.properties) == 1
    assert entity_type.spec.properties[0].name == "type"
    assert entity_type.spec.properties[0].prop_type == "String"
    assert entity_type.spec.properties[0].optional is True

    semantics = pgs_generate.SchemaSemantics(node_types, edge_types)
    options = semantics.eval_node_type("EntityType")
    assert len(options) == 1
    assert options[0].record_spec.optional == {"type": "STRING"}

    rng = pgs_generate.random.Random(23)
    nodes, edges, _, _ = pgs_generate.generate_instances(
        graph_types[0], node_types, edge_types, 1, rng
    )
    assert nodes
    assert not edges


def test_list_property_types_parse_and_generate():
    node_types, edge_types, graph_types = pgs_generate.parse_schema(LIST_PROPERTY_SCHEMA)

    list_node = node_types["ListNodeType"]
    prop_types = {prop.name: prop.prop_type for prop in list_node.spec.properties}
    assert prop_types == {
        "propKeyStrings": "LIST<STRING>",
        "propKeyInts": "LIST<INT>",
        "propKeyFloats": "LIST<FLOAT>",
        "propKeyBooleans": "LIST<BOOLEAN>",
        "propKeyDates": "LIST<DATE>",
        "propKeyDatetimes": "LIST<DATETIME>",
    }

    semantics = pgs_generate.SchemaSemantics(node_types, edge_types)
    options = semantics.eval_node_type("ListNodeType")
    assert len(options) == 1
    assert options[0].record_spec.required == prop_types

    rng = pgs_generate.random.Random(29)
    nodes, edges, _, _ = pgs_generate.generate_instances(
        graph_types[0], node_types, edge_types, 1, rng
    )

    assert nodes
    assert not edges
    for node in nodes:
        for key, prop_type in prop_types.items():
            assert key in node.props
            assert isinstance(node.props[key], list)
            assert node.props[key]
            assert _is_value_of_type(node.props[key], prop_type)


def test_list_property_rejects_unknown_item_type():
    schema = """
    CREATE GRAPH TYPE G STRICT {
      (N: Person {tags LIST<UUID>})
    };
    """

    with pytest.raises(ValueError, match="Unsupported LIST item type UUID"):
        pgs_generate.parse_schema(schema)


def test_type_ref_prefix_parses_and_generates():
    node_types, edge_types, graph_types = pgs_generate.parse_schema(
        TYPE_REF_SCHEMA,
        type_ref_prefix="@",
    )
    graph_type = graph_types[0]
    rng = pgs_generate.random.Random(31)

    nodes, edges, _, _ = pgs_generate.generate_instances(graph_type, node_types, edge_types, 1, rng)
    semantics = pgs_generate.SchemaSemantics(node_types, edge_types)
    nodes_by_id = {n.node_id: n for n in nodes}

    assert len(nodes) == 30
    assert len(edges) == 30

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
    subprocess.run(
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


def test_cli_type_ref_prefix_generates_graphml(tmp_path):
    schema_path = tmp_path / "schema.pgs"
    schema_path.write_text(TYPE_REF_SCHEMA, encoding="utf-8")
    out_path = tmp_path / "out.graphml"
    subprocess.run(
        [
            sys.executable,
            "pgs_generate.py",
            str(schema_path),
            "1",
            "--type-ref-prefix",
            "@",
            "-o",
            str(out_path),
        ],
        cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        check=True,
        capture_output=True,
        text=True,
    )

    tree = ET.parse(out_path)
    root = tree.getroot()
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
    graph = root.find("g:graph", ns)
    assert graph is not None
    assert graph.find("g:node", ns) is not None
    assert graph.find("g:edge", ns) is not None


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


def test_apoc_graphml_declares_property_types():
    node_types, edge_types, graph_types = pgs_generate.parse_schema(APOC_TYPED_GRAPHML_SCHEMA)
    graph_type = graph_types[0]
    rng = pgs_generate.random.Random(41)

    nodes, edges, node_prop_names, edge_prop_names = pgs_generate.generate_instances(
        graph_type, node_types, edge_types, 1, rng
    )
    tree = pgs_generate.build_graphml(nodes, edges, node_prop_names, edge_prop_names)
    root = tree.getroot()
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}

    node_key_elements = root.findall("g:key[@for='node']", ns)
    node_keys = {
        key_el.attrib["attr.name"]: key_el.attrib.get("attr.type") for key_el in node_key_elements
    }
    node_key_ids = {key_el.attrib["attr.name"]: key_el.attrib["id"] for key_el in node_key_elements}
    edge_keys = {
        key_el.attrib["attr.name"]: key_el.attrib.get("attr.type")
        for key_el in root.findall("g:key[@for='edge']", ns)
    }

    assert node_keys["labels"] == "string"
    assert node_keys["propkeyInteger"] in {"int", "long"}
    assert node_keys["propKeyLong"] in {"int", "long"}
    assert node_keys["propKeyDouble"] == "double"
    assert node_keys["propKeyBoolean"] == "boolean"
    assert node_keys["propKeyList"] == "string"
    assert edge_keys["label"] == "string"

    graph = root.find("g:graph", ns)
    node = graph.find("g:node", ns)
    bool_data = node.find("g:data[@key='propKeyBoolean']", ns)
    assert bool_data is not None
    assert bool_data.text in {"true", "false"}
    list_data = node.find(f"g:data[@key='{node_key_ids['propKeyList']}']", ns)
    assert list_data is not None
    assert json.loads(list_data.text) == nodes[0].props["propKeyList"]


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


def test_nonconforming_extra_label_and_prop_mutations():
    node_types, edge_types, graph_types = pgs_generate.parse_schema(MUTATION_SCHEMA)
    graph_type = graph_types[0]
    rng = pgs_generate.random.Random(23)
    mutation_plan = pgs_generate.MutationPlan(
        per_type={
            "Person": pgs_generate.MutationRates(
                extra_label_probability=1.0,
                extra_property_probability=1.0,
            ),
            "Knows": pgs_generate.MutationRates(
                extra_label_probability=1.0,
                extra_property_probability=1.0,
            ),
        }
    )

    nodes, edges, _, _ = pgs_generate.generate_instances(
        graph_type,
        node_types,
        edge_types,
        1,
        rng,
        mutation_plan=mutation_plan,
    )

    assert nodes
    assert edges
    assert all(any(label.startswith("MutatedLabel") for label in node.labels) for node in nodes)
    assert all(any(key.startswith("mutated_prop_") for key in node.props) for node in nodes)
    assert all(any(label.startswith("MutatedLabel") for label in edge.labels) for edge in edges)
    assert all(any(key.startswith("mutated_prop_") for key in edge.props) for edge in edges)


def test_nonconforming_invalid_optional_property_mutation():
    node_types, edge_types, graph_types = pgs_generate.parse_schema(MUTATION_SCHEMA)
    graph_type = graph_types[0]
    rng = pgs_generate.random.Random(24)
    mutation_plan = pgs_generate.MutationPlan(
        per_type={
            "Person": pgs_generate.MutationRates(
                invalid_optional_property_probability=1.0,
            ),
            "Knows": pgs_generate.MutationRates(
                invalid_optional_property_probability=1.0,
            ),
        }
    )

    nodes, edges, _, _ = pgs_generate.generate_instances(
        graph_type,
        node_types,
        edge_types,
        1,
        rng,
        mutation_plan=mutation_plan,
    )

    assert all("nick" in node.props for node in nodes)
    assert all(not _is_value_of_type(node.props["nick"], "STRING") for node in nodes)
    assert all("weight" in edge.props for edge in edges)
    assert all(not _is_value_of_type(edge.props["weight"], "INT") for edge in edges)


def test_nonconforming_wrong_property_datatype_mutation():
    node_types, edge_types, graph_types = pgs_generate.parse_schema(MUTATION_SCHEMA)
    graph_type = graph_types[0]
    rng = pgs_generate.random.Random(28)
    mutation_plan = pgs_generate.MutationPlan(
        per_type={
            "Person": pgs_generate.MutationRates(
                wrong_property_datatype_probability=1.0,
            ),
            "Knows": pgs_generate.MutationRates(
                wrong_property_datatype_probability=1.0,
            ),
        }
    )

    nodes, edges, _, _ = pgs_generate.generate_instances(
        graph_type,
        node_types,
        edge_types,
        1,
        rng,
        mutation_plan=mutation_plan,
    )

    assert all("id" in node.props for node in nodes)
    assert all(not _is_value_of_type(node.props["id"], "INT") for node in nodes)
    assert all("since" in edge.props for edge in edges)
    assert all(not _is_value_of_type(edge.props["since"], "DATE") for edge in edges)


def test_nonconforming_missing_required_node_fields():
    schema = """
    CREATE GRAPH TYPE G STRICT {
      (Person: Person {id INT})
    };
    """
    node_types, edge_types, graph_types = pgs_generate.parse_schema(schema)
    graph_type = graph_types[0]
    rng = pgs_generate.random.Random(25)
    mutation_plan = pgs_generate.MutationPlan(
        per_type={
            "Person": pgs_generate.MutationRates(
                missing_required_property_probability=1.0,
                missing_required_label_probability=1.0,
            )
        }
    )

    nodes, edges, _, _ = pgs_generate.generate_instances(
        graph_type,
        node_types,
        edge_types,
        1,
        rng,
        mutation_plan=mutation_plan,
    )
    semantics = pgs_generate.SchemaSemantics(node_types, edge_types)
    option = semantics.eval_node_type("Person")[0]

    assert nodes
    assert not edges
    assert all(not pgs_generate._record_conforms(node.props, option.record_spec) for node in nodes)
    assert all(not pgs_generate._labels_conform(set(node.labels), option) for node in nodes)


def test_nonconforming_missing_required_edge_fields():
    node_types, edge_types, graph_types = pgs_generate.parse_schema(MUTATION_SCHEMA)
    graph_type = graph_types[0]
    rng = pgs_generate.random.Random(26)
    mutation_plan = pgs_generate.MutationPlan(
        per_type={
            "Knows": pgs_generate.MutationRates(
                missing_required_property_probability=1.0,
                missing_required_label_probability=1.0,
            )
        }
    )

    nodes, edges, _, _ = pgs_generate.generate_instances(
        graph_type,
        node_types,
        edge_types,
        1,
        rng,
        mutation_plan=mutation_plan,
    )
    semantics = pgs_generate.SchemaSemantics(node_types, edge_types)
    option = semantics.eval_edge_type("Knows")[0].edge

    assert nodes
    assert edges
    assert all(not pgs_generate._record_conforms(edge.props, option.record_spec) for edge in edges)
    assert all(not pgs_generate._labels_conform(set(edge.labels), option) for edge in edges)


def test_nonconforming_typo_label_and_property_key_mutations():
    node_types, edge_types, graph_types = pgs_generate.parse_schema(TYPO_SCHEMA)
    graph_type = graph_types[0]
    rng = pgs_generate.random.Random(29)
    mutation_plan = pgs_generate.MutationPlan(
        per_type={
            "Person": pgs_generate.MutationRates(
                typo_label_probability=1.0,
                typo_property_key_probability=1.0,
            ),
            "Knows": pgs_generate.MutationRates(
                typo_label_probability=1.0,
                typo_property_key_probability=1.0,
            ),
        }
    )

    nodes, edges, _, _ = pgs_generate.generate_instances(
        graph_type,
        node_types,
        edge_types,
        1,
        rng,
        mutation_plan=mutation_plan,
    )

    assert nodes
    assert edges
    assert all(node.labels == [label] and label != "Person" for node in nodes for label in node.labels)
    assert all("id" not in node.props and len(node.props) == 1 for node in nodes)
    assert all(edge.labels == [label] and label != "KNOWS" for edge in edges for label in edge.labels)
    assert all("since" not in edge.props and len(edge.props) == 1 for edge in edges)


def test_nonconforming_fresh_nodes_and_edges():
    node_types, edge_types, graph_types = pgs_generate.parse_schema(MUTATION_SCHEMA)
    graph_type = graph_types[0]
    rng = pgs_generate.random.Random(27)
    mutation_plan = pgs_generate.MutationPlan(
        defaults=pgs_generate.MutationRates(
            fresh_node_probability=1.0,
            fresh_edge_probability=1.0,
        )
    )

    nodes, edges, _, _ = pgs_generate.generate_instances(
        graph_type,
        node_types,
        edge_types,
        1,
        rng,
        mutation_plan=mutation_plan,
    )

    assert len(nodes) > 10
    assert len(edges) > 15
    assert any(node.type_name == "__fresh_node__" for node in nodes)
    assert any(edge.type_name == "__fresh_edge__" for edge in edges)


def test_mutation_report_collects_summary_and_object_details():
    node_types, edge_types, graph_types = pgs_generate.parse_schema(MUTATION_SCHEMA)
    graph_type = graph_types[0]
    rng = pgs_generate.random.Random(30)
    mutation_plan = pgs_generate.MutationPlan(
        per_type={
            "Person": pgs_generate.MutationRates(
                extra_label_probability=1.0,
                wrong_property_datatype_probability=1.0,
            ),
            "Knows": pgs_generate.MutationRates(
                typo_label_probability=1.0,
            ),
        }
    )
    mutation_report = pgs_generate.MutationReport()

    nodes, edges, _, _ = pgs_generate.generate_instances(
        graph_type,
        node_types,
        edge_types,
        1,
        rng,
        mutation_plan=mutation_plan,
        mutation_report=mutation_report,
    )

    payload = mutation_report.to_dict(
        nodes,
        edges,
        meta={"schema": "memory.pgs", "graph_type": graph_type.name, "seed": 30, "scale": 1},
    )

    assert payload["meta"]["graph_type"] == graph_type.name
    assert payload["summary"]["nodes_total"] == len(nodes)
    assert payload["summary"]["edges_total"] == len(edges)
    assert payload["summary"]["mutated_nodes"] > 0
    assert payload["summary"]["mutated_edges"] > 0
    assert payload["summary"]["by_kind"]["extra_label"] > 0
    assert payload["summary"]["by_kind"]["wrong_property_datatype"] > 0
    assert payload["summary"]["by_kind"]["typo_label"] > 0
    assert any(
        obj["kind"] == "node"
        and any(mutation["kind"] == "wrong_property_datatype" for mutation in obj["mutations"])
        for obj in payload["objects"]
    )
    assert any(
        obj["kind"] == "edge"
        and "source" in obj
        and "target" in obj
        and any(mutation["kind"] == "typo_label" for mutation in obj["mutations"])
        for obj in payload["objects"]
    )


def test_load_mutation_plan_merges_defaults_types_and_cli(tmp_path):
    config_path = tmp_path / "mutations.json"
    config_path.write_text(
        json.dumps(
            {
                "defaults": {"extra_label_probability": 0.25},
                "types": {"Person": {"missing_required_label_probability": 1.0}},
            }
        ),
        encoding="utf-8",
    )

    mutation_plan = pgs_generate.load_mutation_plan(
        config_path,
        {
            "extra_property_probability": 0.5,
            "typo_label_probability": 0.75,
        },
    )

    default_rates = mutation_plan.for_type("Other")
    person_rates = mutation_plan.for_type("Person")
    assert default_rates.extra_label_probability == 0.25
    assert default_rates.extra_property_probability == 0.5
    assert default_rates.typo_label_probability == 0.75
    assert person_rates.extra_label_probability == 0.25
    assert person_rates.extra_property_probability == 0.5
    assert person_rates.typo_label_probability == 0.75
    assert person_rates.missing_required_label_probability == 1.0


def test_cli_mutation_flags_generate_nonconforming_graphml(tmp_path):
    schema = """
    CREATE GRAPH TYPE G STRICT {
      (Person: Person {id INT})
    };
    """
    schema_path = tmp_path / "schema.pgs"
    schema_path.write_text(schema, encoding="utf-8")
    out_path = tmp_path / "out.graphml"

    subprocess.run(
        [
            sys.executable,
            "pgs_generate.py",
            str(schema_path),
            "1",
            "--mutation-extra-label-prob",
            "1.0",
            "--mutation-extra-prop-prob",
            "1.0",
            "-o",
            str(out_path),
        ],
        cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        check=True,
        capture_output=True,
        text=True,
    )

    tree = ET.parse(out_path)
    root = tree.getroot()
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
    graph = root.find("g:graph", ns)
    nodes = graph.findall("g:node", ns)
    assert nodes

    keys = {}
    for key_el in root.findall("g:key", ns):
        keys[key_el.attrib["id"]] = key_el.attrib.get("attr.name", "")

    for node in nodes:
        labels = (node.attrib.get("labels") or "").split(":")
        labels = [label for label in labels if label]
        assert any(label.startswith("MutatedLabel") for label in labels)
        assert any(
            keys.get(data.attrib.get("key", ""), "").startswith("mutated_prop_")
            for data in node.findall("g:data", ns)
        )


def test_cli_typo_and_wrong_datatype_flags(tmp_path):
    schema_path = tmp_path / "schema.pgs"
    schema_path.write_text(TYPO_SCHEMA, encoding="utf-8")
    out_path = tmp_path / "out.graphml"

    subprocess.run(
        [
            sys.executable,
            "pgs_generate.py",
            str(schema_path),
            "1",
            "--mutation-wrong-prop-datatype-prob",
            "1.0",
            "--mutation-typo-label-prob",
            "1.0",
            "--mutation-typo-prop-key-prob",
            "1.0",
            "-o",
            str(out_path),
        ],
        cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        check=True,
        capture_output=True,
        text=True,
    )

    tree = ET.parse(out_path)
    root = tree.getroot()
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
    graph = root.find("g:graph", ns)
    nodes = graph.findall("g:node", ns)
    edges = graph.findall("g:edge", ns)
    assert nodes
    assert edges

    keys = {}
    for key_el in root.findall("g:key", ns):
        keys[key_el.attrib["id"]] = key_el.attrib.get("attr.name", "")

    for node in nodes:
        labels = (node.attrib.get("labels") or "").split(":")
        labels = [label for label in labels if label]
        assert labels and labels[0] != "Person"
        attr_names = [keys.get(data.attrib.get("key", ""), "") for data in node.findall("g:data", ns)]
        assert "id" not in attr_names

    for edge in edges:
        assert edge.attrib.get("label") != "KNOWS"
        attr_names = [keys.get(data.attrib.get("key", ""), "") for data in edge.findall("g:data", ns)]
        assert "since" not in attr_names


def test_cli_writes_mutation_report_json(tmp_path):
    schema_path = tmp_path / "schema.pgs"
    schema_path.write_text(MUTATION_SCHEMA, encoding="utf-8")
    out_path = tmp_path / "out.graphml"
    report_path = tmp_path / "mutation-report.json"

    subprocess.run(
        [
            sys.executable,
            "pgs_generate.py",
            str(schema_path),
            "1",
            "--mutation-extra-label-prob",
            "1.0",
            "--mutation-report",
            str(report_path),
            "-o",
            str(out_path),
        ],
        cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        check=True,
        capture_output=True,
        text=True,
    )

    assert out_path.exists()
    assert report_path.exists()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["meta"]["schema"] == str(schema_path)
    assert payload["meta"]["graph_type"] == "G"
    assert payload["summary"]["nodes_total"] > 0
    assert payload["summary"]["mutated_nodes"] > 0
    assert payload["summary"]["by_kind"]["extra_label"] > 0
    assert any(obj["kind"] == "node" for obj in payload["objects"])
