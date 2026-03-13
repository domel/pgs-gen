#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import random
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass


KEYWORDS = {
    "CREATE",
    "NODE",
    "EDGE",
    "GRAPH",
    "TYPE",
    "ABSTRACT",
    "STRICT",
    "LOOSE",
    "IMPORTS",
    "OPEN",
    "OPTIONAL",
}

MUTATION_FIELDS = (
    "fresh_node_probability",
    "fresh_edge_probability",
    "extra_label_probability",
    "extra_property_probability",
    "invalid_optional_property_probability",
    "wrong_property_datatype_probability",
    "missing_required_property_probability",
    "missing_required_label_probability",
    "typo_label_probability",
    "typo_property_key_probability",
)


class Token:
    def __init__(self, kind, value, pos):
        self.kind = kind
        self.value = value
        self.pos = pos


class Tokenizer:
    def __init__(self, text):
        self.text = text
        self.i = 0

    def _peek(self):
        if self.i >= len(self.text):
            return ""
        return self.text[self.i]

    def _advance(self):
        ch = self._peek()
        self.i += 1
        return ch

    def _skip_ws(self):
        while self._peek() and self._peek().isspace():
            self.i += 1

    def next_token(self):
        self._skip_ws()
        pos = self.i
        ch = self._peek()
        if not ch:
            return Token("EOF", "", pos)
        if ch == "-":
            nxt = self.text[self.i + 1] if self.i + 1 < len(self.text) else ""
            if not nxt or nxt.isspace() or nxt in "[](){}:,;>":
                self.i += 1
                return Token("SYM", "-", pos)
        if ch.isalnum() or ch in "_-":
            start = self.i
            while self._peek() and (self._peek().isalnum() or self._peek() in "_-"):
                self.i += 1
            value = self.text[start:self.i]
            upper = value.upper()
            if upper in KEYWORDS:
                return Token("KW", upper, pos)
            return Token("IDENT", value, pos)
        if ch in "()[]{}:,|&?-;>" or ch == "-":
            self.i += 1
            return Token("SYM", ch, pos)
        raise ValueError(f"Unexpected character '{ch}' at {pos}")


class Parser:
    def __init__(self, text):
        self.tokens = []
        t = Tokenizer(text)
        while True:
            tok = t.next_token()
            self.tokens.append(tok)
            if tok.kind == "EOF":
                break
        self.i = 0

    def _peek(self):
        return self.tokens[self.i]

    def _next(self):
        tok = self.tokens[self.i]
        self.i += 1
        return tok

    def _match_kw(self, value):
        tok = self._peek()
        if tok.kind == "KW" and tok.value == value:
            self._next()
            return True
        return False

    def _match_sym(self, value):
        tok = self._peek()
        if tok.kind == "SYM" and tok.value == value:
            self._next()
            return True
        return False

    def _expect_kw(self, value):
        if not self._match_kw(value):
            tok = self._peek()
            raise ValueError(f"Expected keyword {value} at {tok.pos}")

    def _expect_sym(self, value):
        if not self._match_sym(value):
            tok = self._peek()
            raise ValueError(f"Expected symbol '{value}' at {tok.pos}")

    def _expect_ident(self):
        tok = self._peek()
        if tok.kind == "IDENT":
            self._next()
            return tok.value
        raise ValueError(f"Expected identifier at {tok.pos}")

    def parse(self):
        statements = []
        while self._peek().kind != "EOF":
            if self._match_sym(";"):
                continue
            if self._match_kw("CREATE"):
                statements.append(self._parse_create())
            else:
                tok = self._peek()
                raise ValueError(f"Unexpected token {tok.value} at {tok.pos}")
        return statements

    def _parse_create(self):
        if self._match_kw("NODE"):
            self._expect_kw("TYPE")
            abstract = self._match_kw("ABSTRACT")
            node_type = self._parse_node_type(abstract)
            return ("node", node_type)
        if self._match_kw("EDGE"):
            self._expect_kw("TYPE")
            abstract = self._match_kw("ABSTRACT")
            edge_type = self._parse_edge_type(abstract)
            return ("edge", edge_type)
        if self._match_kw("GRAPH"):
            self._expect_kw("TYPE")
            graph_type = self._parse_graph_type()
            return ("graph", graph_type)
        tok = self._peek()
        raise ValueError(f"Expected NODE/EDGE/GRAPH at {tok.pos}")

    def _parse_graph_type(self):
        name = self._expect_ident()
        if self._match_kw("STRICT"):
            form = "STRICT"
        elif self._match_kw("LOOSE"):
            form = "LOOSE"
        else:
            tok = self._peek()
            raise ValueError(f"Expected STRICT/LOOSE at {tok.pos}")
        elements = self._parse_graph_def()
        return GraphType(name, form, elements)

    def _parse_graph_def(self):
        if self._match_kw("IMPORTS"):
            _ = self._expect_ident()
        self._expect_sym("{")
        elements = []
        if not self._match_sym("}"):
            elements.append(self._parse_element_type())
            while self._match_sym(","):
                elements.append(self._parse_element_type())
            self._expect_sym("}")
        return elements

    def _parse_element_type(self):
        tok = self._peek()
        if tok.kind == "SYM" and tok.value == "(":
            # Lookahead to distinguish node vs edge type
            self._next()  # consume '('
            next_tok = self._peek()
            # restore by moving back
            self.i -= 1
            if next_tok.kind == "IDENT":
                return self._parse_node_type()
            return self._parse_edge_type()
        if tok.kind == "IDENT":
            name = self._expect_ident()
            return name
        raise ValueError(f"Unexpected element type at {tok.pos}")

    def _parse_node_type(self, abstract=False):
        self._expect_sym("(")
        name = self._expect_ident()
        spec = self._parse_label_property_spec()
        self._expect_sym(")")
        return NodeType(name, spec, abstract)

    def _parse_edge_type(self, abstract=False):
        left = self._parse_endpoint_type()
        self._expect_sym("-")
        middle_name, middle_spec = self._parse_middle_type()
        self._expect_sym("-")
        if not self._match_sym(">"):
            tok = self._peek()
            raise ValueError(f"Expected '>' at {tok.pos}")
        right = self._parse_endpoint_type()
        name = middle_name
        if not name:
            name = _label_spec_primary(middle_spec.labels_ast)
        if not name:
            name = "EdgeType"
        return EdgeType(name, middle_spec, left, right, abstract)

    def _parse_middle_type(self):
        self._expect_sym("[")
        name = None
        if self._peek().kind == "IDENT":
            name = self._expect_ident()
        spec = self._parse_label_property_spec()
        self._expect_sym("]")
        return name, spec

    def _parse_endpoint_type(self):
        self._expect_sym("(")
        spec = self._parse_label_property_spec()
        self._expect_sym(")")
        return spec

    def _parse_label_property_spec(self):
        labels_ast = None
        label_open = False
        props = []
        prop_open = False
        label_spec_present = False
        prop_spec_present = False
        if self._match_sym(":"):
            label_spec_present = True
            labels_ast = self._parse_label_spec()
        if self._match_kw("OPEN"):
            label_open = True
        if self._match_sym("{"):
            prop_spec_present = True
            if self._match_sym("}"):
                return LabelPropSpec(
                    labels_ast, label_open, props, prop_open, label_spec_present, prop_spec_present
                )
            if self._match_kw("OPEN"):
                prop_open = True
                self._expect_sym("}")
                return LabelPropSpec(
                    labels_ast, label_open, props, prop_open, label_spec_present, prop_spec_present
                )
            props = self._parse_properties()
            if self._match_sym(","):
                if self._match_kw("OPEN"):
                    prop_open = True
            self._expect_sym("}")
        return LabelPropSpec(
            labels_ast, label_open, props, prop_open, label_spec_present, prop_spec_present
        )

    def _parse_properties(self):
        props = []
        props.append(self._parse_property())
        while self._peek().kind == "SYM" and self._peek().value == ",":
            next_tok = self.tokens[self.i + 1] if self.i + 1 < len(self.tokens) else None
            if next_tok and next_tok.kind == "KW" and next_tok.value == "OPEN":
                break
            self._match_sym(",")
            props.append(self._parse_property())
        return props

    def _parse_property(self):
        optional = self._match_kw("OPTIONAL")
        key = self._expect_ident()
        prop_type = self._expect_ident()
        return Property(key, prop_type, optional)

    def _parse_label_spec(self):
        return self._parse_label_or()

    def _parse_label_or(self):
        node = self._parse_label_and()
        while self._match_sym("|"):
            right = self._parse_label_and()
            node = ("or", node, right)
        return node

    def _parse_label_and(self):
        node = self._parse_label_postfix()
        while self._match_sym("&"):
            right = self._parse_label_postfix()
            node = ("and", node, right)
        return node

    def _parse_label_postfix(self):
        node = self._parse_label_atom()
        if self._match_sym("?"):
            node = ("optional", node)
        return node

    def _parse_label_atom(self):
        tok = self._peek()
        if self._match_sym(":"):
            tok = self._peek()
            if tok.kind == "IDENT":
                return self._expect_ident()
            raise ValueError(f"Expected label at {tok.pos}")
        if self._match_sym("("):
            inner = self._parse_label_spec()
            self._expect_sym(")")
            return inner
        if self._match_sym("["):
            inner = self._parse_label_spec()
            self._expect_sym("]")
            return inner
        if tok.kind == "IDENT":
            return self._expect_ident()
        raise ValueError(f"Expected label at {tok.pos}")


class Property:
    def __init__(self, name, prop_type, optional):
        self.name = name
        self.prop_type = prop_type
        self.optional = optional


class LabelPropSpec:
    def __init__(
        self, labels_ast, label_open, properties, prop_open, label_spec_present, prop_spec_present
    ):
        self.labels_ast = labels_ast
        self.label_open = label_open
        self.properties = properties
        self.prop_open = prop_open
        self.label_spec_present = label_spec_present
        self.prop_spec_present = prop_spec_present


class NodeType:
    def __init__(self, name, spec, abstract=False):
        self.name = name
        self.spec = spec
        self.abstract = abstract


class EdgeType:
    def __init__(self, name, spec, left_spec, right_spec, abstract=False):
        self.name = name
        self.spec = spec
        self.left_spec = left_spec
        self.right_spec = right_spec
        self.abstract = abstract


class GraphType:
    def __init__(self, name, form, elements):
        self.name = name
        self.form = form
        self.elements = elements


@dataclass(frozen=True)
class MutationRates:
    fresh_node_probability: float = 0.0
    fresh_edge_probability: float = 0.0
    extra_label_probability: float = 0.0
    extra_property_probability: float = 0.0
    invalid_optional_property_probability: float = 0.0
    wrong_property_datatype_probability: float = 0.0
    missing_required_property_probability: float = 0.0
    missing_required_label_probability: float = 0.0
    typo_label_probability: float = 0.0
    typo_property_key_probability: float = 0.0


class MutationPlan:
    def __init__(self, defaults=None, per_type=None):
        self.defaults = defaults or MutationRates()
        self.per_type = per_type or {}

    def for_type(self, type_name):
        return self.per_type.get(type_name, self.defaults)


class MutationReport:
    def __init__(self):
        self._objects = []
        self._objects_by_id = {}
        self._by_kind = {}

    def record_object_mutations(
        self,
        object_id,
        object_kind,
        type_name,
        mutations,
        source=None,
        target=None,
    ):
        if not mutations:
            return

        entry = self._objects_by_id.get(object_id)
        if entry is None:
            entry = {
                "id": object_id,
                "kind": object_kind,
                "type": type_name,
                "mutations": [],
            }
            if source is not None:
                entry["source"] = source
            if target is not None:
                entry["target"] = target
            self._objects.append(entry)
            self._objects_by_id[object_id] = entry

        entry["mutations"].extend(mutations)
        for mutation in mutations:
            kind = mutation["kind"]
            self._by_kind[kind] = self._by_kind.get(kind, 0) + 1

    def to_dict(self, nodes, edges, meta=None):
        meta_payload = dict(meta or {})
        mutated_nodes = sum(1 for entry in self._objects if entry["kind"] == "node")
        mutated_edges = sum(1 for entry in self._objects if entry["kind"] == "edge")
        return {
            "meta": meta_payload,
            "summary": {
                "nodes_total": len(nodes),
                "edges_total": len(edges),
                "mutated_nodes": mutated_nodes,
                "mutated_edges": mutated_edges,
                "by_kind": dict(sorted(self._by_kind.items())),
            },
            "objects": self._objects,
        }

    def write_json(self, path, nodes, edges, meta=None):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(nodes, edges, meta=meta), handle, ensure_ascii=False, indent=2)
            handle.write("\n")


class NodeInstance:
    def __init__(self, node_id, labels, props, type_name, schema_labels=None, schema_props=None):
        self.node_id = node_id
        self.labels = labels
        self.props = props
        self.type_name = type_name
        self.schema_labels = list(schema_labels) if schema_labels is not None else list(labels)
        self.schema_props = dict(schema_props) if schema_props is not None else dict(props)


class EdgeInstance:
    def __init__(self, edge_id, source, target, labels, props, type_name):
        self.edge_id = edge_id
        self.source = source
        self.target = target
        self.labels = labels
        self.props = props
        self.type_name = type_name


def parse_schema(text):
    parser = Parser(text)
    statements = parser.parse()
    node_types = {}
    edge_types = {}
    graph_types = []
    for kind, obj in statements:
        if kind == "node":
            node_types[obj.name] = obj
        elif kind == "edge":
            edge_types[obj.name] = obj
        elif kind == "graph":
            graph_types.append(obj)
    for graph_type in graph_types:
        for element in graph_type.elements:
            if isinstance(element, NodeType):
                node_types.setdefault(element.name, element)
            elif isinstance(element, EdgeType):
                edge_types.setdefault(element.name, element)
    return node_types, edge_types, graph_types


def eval_label_spec(ast, rng):
    if ast is None:
        return []
    if isinstance(ast, str):
        return [ast]
    kind = ast[0]
    if kind == "and":
        left = eval_label_spec(ast[1], rng)
        right = eval_label_spec(ast[2], rng)
        return left + [x for x in right if x not in left]
    if kind == "or":
        return eval_label_spec(ast[1], rng) if rng.random() < 0.5 else eval_label_spec(ast[2], rng)
    if kind == "optional":
        return eval_label_spec(ast[1], rng) if rng.random() < 0.5 else []
    return []


def _label_spec_primary(ast):
    if ast is None:
        return None
    if isinstance(ast, str):
        return ast
    kind = ast[0]
    if kind in {"and", "or"}:
        return _label_spec_primary(ast[1]) or _label_spec_primary(ast[2])
    if kind == "optional":
        return _label_spec_primary(ast[1])
    return None


def _canonical_prop_type(prop_type):
    t = prop_type.strip().upper()
    if t in {"INT32", "INT64", "INT", "LONG"}:
        return "INT"
    if t in {"DOUBLE", "FLOAT", "DECIMAL"}:
        return "FLOAT"
    if t in {"BOOLEAN", "BOOL"}:
        return "BOOLEAN"
    if t in {"DATETIME", "TIMESTAMP"}:
        return "DATETIME"
    if t == "DATE":
        return "DATE"
    if t == "STRING":
        return "STRING"
    return t


def _collect_label_names(ast):
    if ast is None:
        return set()
    if isinstance(ast, str):
        return {ast}
    kind = ast[0]
    if kind in {"and", "or"}:
        return _collect_label_names(ast[1]) | _collect_label_names(ast[2])
    if kind == "optional":
        return _collect_label_names(ast[1])
    return set()


def _validate_probability(name, value, context):
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{context}: {name} must be between 0.0 and 1.0")


def _mutation_rates_from_mapping(mapping, base=None, context="mutation config"):
    values = dict(vars(base or MutationRates()))
    for field in MUTATION_FIELDS:
        if field in mapping and mapping[field] is not None:
            values[field] = float(mapping[field])
    rates = MutationRates(**values)
    for field in MUTATION_FIELDS:
        _validate_probability(field, getattr(rates, field), context)
    return rates


def load_mutation_plan(path=None, cli_overrides=None):
    payload = {}
    if path:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("Mutation config must be a JSON object")

    raw_defaults = payload.get("defaults")
    if raw_defaults is None:
        raw_defaults = payload.get("global", {})
    if raw_defaults is None:
        raw_defaults = {}
    if not isinstance(raw_defaults, dict):
        raise ValueError("Mutation config 'defaults' must be an object")

    defaults = _mutation_rates_from_mapping(raw_defaults, context="mutation defaults")
    if cli_overrides:
        defaults = _mutation_rates_from_mapping(cli_overrides, base=defaults, context="CLI mutations")

    raw_types = payload.get("types", {})
    if raw_types is None:
        raw_types = {}
    if not isinstance(raw_types, dict):
        raise ValueError("Mutation config 'types' must be an object")

    per_type = {}
    for type_name, type_mapping in raw_types.items():
        if not isinstance(type_mapping, dict):
            raise ValueError(f"Mutation config for type {type_name} must be an object")
        per_type[type_name] = _mutation_rates_from_mapping(
            type_mapping,
            base=defaults,
            context=f"mutation type {type_name}",
        )

    return MutationPlan(defaults, per_type)


class RecordSpec:
    def __init__(self, required=None, optional=None, open=False):
        self.required = required or {}
        self.optional = optional or {}
        self.open = open


class BaseOption:
    def __init__(self, labels, record_spec, label_open=False):
        self.labels = set(labels)
        self.record_spec = record_spec
        self.label_open = label_open


class EdgeOption:
    def __init__(self, source, edge, target):
        self.source = source
        self.edge = edge
        self.target = target


def _empty_record_spec():
    return RecordSpec({}, {}, False)


def _record_spec_from_properties(properties, prop_open):
    required = {}
    optional = {}
    for prop in properties:
        canon = _canonical_prop_type(prop.prop_type)
        if prop.optional:
            optional[prop.name] = canon
        else:
            required[prop.name] = canon
    return RecordSpec(required, optional, prop_open)


def _combine_record_specs(left, right):
    required = {}
    optional = {}
    for key in set(left.required) | set(left.optional) | set(right.required) | set(right.optional):
        l_type = left.required.get(key) or left.optional.get(key)
        r_type = right.required.get(key) or right.optional.get(key)
        if l_type and r_type and l_type != r_type:
            return None
        prop_type = l_type or r_type
        if key in left.required or key in right.required:
            required[key] = prop_type
        else:
            optional[key] = prop_type
    return RecordSpec(required, optional, left.open or right.open)


def _combine_base_options(left, right):
    record_spec = _combine_record_specs(left.record_spec, right.record_spec)
    if record_spec is None:
        return None
    return BaseOption(left.labels | right.labels, record_spec, left.label_open or right.label_open)


def _value_conforms(value, prop_type):
    t = _canonical_prop_type(prop_type)
    if t == "INT":
        if isinstance(value, bool):
            return False
        if isinstance(value, int):
            return True
        try:
            int(value)
            return True
        except (ValueError, TypeError):
            return False
    if t == "FLOAT":
        if isinstance(value, bool):
            return False
        if isinstance(value, (int, float)):
            return True
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False
    if t == "BOOLEAN":
        if isinstance(value, bool):
            return True
        return value in {"true", "false"}
    if t == "DATE":
        if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
            return True
        try:
            dt.date.fromisoformat(value)
            return True
        except (ValueError, TypeError):
            return False
    if t == "DATETIME":
        if isinstance(value, dt.datetime):
            return True
        try:
            dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
            return True
        except (ValueError, TypeError):
            return False
    return isinstance(value, str)


def _record_conforms(record, record_spec):
    for key, prop_type in record_spec.required.items():
        if key not in record:
            return False
        if not _value_conforms(record[key], prop_type):
            return False
    for key, prop_type in record.items():
        if key in record_spec.required:
            continue
        if key in record_spec.optional:
            if not _value_conforms(record[key], record_spec.optional[key]):
                return False
        elif not record_spec.open:
            return False
    return True


def _labels_conform(labels, option):
    if option.label_open:
        return option.labels.issubset(labels)
    return labels == option.labels


class SchemaSemantics:
    def __init__(self, node_types, edge_types):
        self.node_types = node_types
        self.edge_types = edge_types
        self._node_cache = {}
        self._edge_cache = {}
        self._node_stack = set()
        self._edge_stack = set()

    def eval_node_type(self, name):
        if name in self._node_cache:
            return self._node_cache[name]
        if name in self._node_stack:
            raise ValueError(f"Cyclic node type reference: {name}")
        node_type = self.node_types.get(name)
        if not node_type:
            return [BaseOption({name}, _empty_record_spec(), False)]
        self._node_stack.add(name)
        options = self._eval_label_prop_spec(node_type.spec, is_node=True, current_type=name)
        self._node_stack.remove(name)
        self._node_cache[name] = options
        return options

    def eval_edge_type(self, name):
        if name in self._edge_cache:
            return self._edge_cache[name]
        if name in self._edge_stack:
            raise ValueError(f"Cyclic edge type reference: {name}")
        edge_type = self.edge_types.get(name)
        if not edge_type:
            base = BaseOption({name}, _empty_record_spec(), False)
            empty = BaseOption(set(), _empty_record_spec(), False)
            return [EdgeOption(empty, base, empty)]
        self._edge_stack.add(name)
        src_options = self._eval_label_prop_spec(edge_type.left_spec, is_node=True)
        tgt_options = self._eval_label_prop_spec(edge_type.right_spec, is_node=True)
        edge_options = self._eval_edge_label_prop_spec(edge_type.spec, current_type=name)
        combined = []
        for src in src_options:
            for tgt in tgt_options:
                for edge_opt in edge_options:
                    new_src = _combine_base_options(src, edge_opt.source)
                    new_tgt = _combine_base_options(tgt, edge_opt.target)
                    if new_src is None or new_tgt is None:
                        continue
                    combined.append(EdgeOption(new_src, edge_opt.edge, new_tgt))
        self._edge_stack.remove(name)
        self._edge_cache[name] = combined
        return combined

    def _eval_label_prop_spec(self, spec, is_node, current_type=None):
        if spec is None:
            return [BaseOption(set(), _empty_record_spec(), True)]
        effective_label_open = spec.label_open or not spec.label_spec_present
        effective_prop_open = spec.prop_open or not spec.prop_spec_present
        options = self._eval_label_expr(spec.labels_ast, is_node, current_type)
        record_spec = _record_spec_from_properties(spec.properties, effective_prop_open)
        out = []
        for opt in options:
            combined = _combine_record_specs(opt.record_spec, record_spec)
            if combined is None:
                continue
            out.append(BaseOption(opt.labels, combined, opt.label_open or effective_label_open))
        return out

    def _eval_label_expr(self, ast, is_node, current_type=None):
        if ast is None:
            return [BaseOption(set(), _empty_record_spec(), False)]
        if isinstance(ast, str):
            if is_node and ast in self.node_types and ast != current_type:
                return self.eval_node_type(ast)
            if (not is_node) and ast in self.edge_types:
                # Edge label expressions are handled elsewhere
                return [BaseOption({ast}, _empty_record_spec(), False)]
            return [BaseOption({ast}, _empty_record_spec(), False)]
        kind = ast[0]
        if kind == "or":
            return self._eval_label_expr(ast[1], is_node, current_type) + self._eval_label_expr(
                ast[2], is_node, current_type
            )
        if kind == "and":
            out = []
            left_opts = self._eval_label_expr(ast[1], is_node, current_type)
            right_opts = self._eval_label_expr(ast[2], is_node, current_type)
            for left in left_opts:
                for right in right_opts:
                    combined = _combine_base_options(left, right)
                    if combined is not None:
                        out.append(combined)
            return out
        if kind == "optional":
            empty = BaseOption(set(), _empty_record_spec(), False)
            return [empty] + self._eval_label_expr(ast[1], is_node, current_type)
        return [BaseOption(set(), _empty_record_spec(), False)]

    def _eval_edge_label_expr(self, ast, current_type=None):
        if ast is None:
            empty = BaseOption(set(), _empty_record_spec(), False)
            return [EdgeOption(empty, empty, empty)]
        if isinstance(ast, str):
            if ast in self.edge_types and ast != current_type:
                return self.eval_edge_type(ast)
            empty = BaseOption(set(), _empty_record_spec(), False)
            edge = BaseOption({ast}, _empty_record_spec(), False)
            return [EdgeOption(empty, edge, empty)]
        kind = ast[0]
        if kind == "or":
            return self._eval_edge_label_expr(ast[1], current_type) + self._eval_edge_label_expr(
                ast[2], current_type
            )
        if kind == "and":
            out = []
            left_opts = self._eval_edge_label_expr(ast[1], current_type)
            right_opts = self._eval_edge_label_expr(ast[2], current_type)
            for left in left_opts:
                for right in right_opts:
                    src = _combine_base_options(left.source, right.source)
                    edge = _combine_base_options(left.edge, right.edge)
                    tgt = _combine_base_options(left.target, right.target)
                    if src is None or edge is None or tgt is None:
                        continue
                    out.append(EdgeOption(src, edge, tgt))
            return out
        if kind == "optional":
            empty = BaseOption(set(), _empty_record_spec(), False)
            return [EdgeOption(empty, empty, empty)] + self._eval_edge_label_expr(
                ast[1], current_type
            )
        return []

    def _eval_edge_label_prop_spec(self, spec, current_type=None):
        edge_opts = self._eval_edge_label_expr(spec.labels_ast, current_type)
        record_spec = _record_spec_from_properties(spec.properties, spec.prop_open)
        out = []
        for opt in edge_opts:
            combined = _combine_record_specs(opt.edge.record_spec, record_spec)
            if combined is None:
                continue
            edge = BaseOption(opt.edge.labels, combined, opt.edge.label_open or spec.label_open)
            out.append(EdgeOption(opt.source, edge, opt.target))
        return out


def _fake_string(prop_name, faker, scale_factor):
    name = prop_name.lower()
    if "gender" in name or "sex" in name:
        return faker.random_element(["male", "female", "non-binary", "other"])
    if "birthdate" in name or "birth_date" in name or name == "dob":
        return faker.date_of_birth(minimum_age=0, maximum_age=100).isoformat()
    if "age" in name:
        return str(faker.random_int(min=0, max=100))
    if "fname" in name or "first_name" in name or "firstname" in name:
        return faker.first_name()
    if "lname" in name or "last_name" in name or "lastname" in name:
        return faker.last_name()
    if "full_name" in name:
        return faker.name()
    if "email" in name:
        return faker.email()
    if "url" in name:
        return faker.url()
    if "uuid" in name:
        return str(faker.uuid4())
    if "ip" in name:
        return faker.ipv4()
    if "lat" in name or "latitude" in name:
        return f"{faker.latitude():.6f}"
    if "lon" in name or "lng" in name or "longitude" in name:
        return f"{faker.longitude():.6f}"
    if "name" in name:
        return faker.name()
    if "title" in name:
        return faker.sentence(nb_words=3).rstrip(".")
    if "description" in name or "desc" in name:
        return faker.sentence(nb_words=8).rstrip(".")
    if "amount" in name or "price" in name or "total" in name:
        return f"{faker.pydecimal(left_digits=4, right_digits=2, positive=True)}"
    if "currency" in name:
        return faker.currency_code()
    if "street" in name or "addr" in name or "address" in name:
        return faker.street_address()
    if "zip" in name or "postal" in name or "postcode" in name:
        return faker.postcode()
    if "company" in name or "org" in name:
        return faker.company()
    if "phone" in name or "tel" in name:
        return faker.phone_number()
    if "path" in name:
        return faker.file_path(depth=2)
    if "username" in name or "user_name" in name:
        return faker.user_name()
    if "domain" in name:
        return faker.domain_name()
    if "owner" in name or "user" in name:
        return faker.user_name()
    if "city" in name:
        return faker.city()
    if "state" in name:
        return faker.state()
    if "country" in name:
        return faker.country()
    if "cc_num" in name or "card" in name:
        return faker.credit_card_number()
    if "id" in name:
        return str(faker.random_int(min=0, max=1000 * scale_factor + 100))
    return faker.word()


def generate_value(prop_type, rng, scale_factor, idx, prop_name, faker=None):
    t = prop_type.upper()
    if t in {"INT", "INT32", "INT64", "LONG"}:
        if faker:
            name = prop_name.lower()
            if "age" in name:
                return faker.random_int(min=0, max=100)
            return faker.random_int(min=0, max=1000 * scale_factor + 100)
        return rng.randint(0, 1000 * scale_factor + 100)
    if t in {"DOUBLE", "FLOAT", "DECIMAL"}:
        return rng.random() * (1000 * scale_factor + 1)
    if t in {"BOOLEAN", "BOOL"}:
        if faker:
            return faker.pybool()
        return rng.random() < 0.5
    if t == "DATE":
        if faker:
            name = prop_name.lower()
            if "birthdate" in name or "birth_date" in name or name == "dob":
                return faker.date_of_birth(minimum_age=0, maximum_age=100).isoformat()
            return faker.date_between(start_date="-20y", end_date="today").isoformat()
        start = dt.date(2000, 1, 1)
        delta = rng.randint(0, 365 * 20)
        return (start + dt.timedelta(days=delta)).isoformat()
    if t in {"DATETIME", "TIMESTAMP"}:
        if faker:
            value = faker.date_time_between(start_date="-5y", end_date="now", tzinfo=dt.timezone.utc)
            return value.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        start = dt.datetime(2000, 1, 1, 0, 0, 0)
        delta = dt.timedelta(seconds=rng.randint(0, 3600 * 24 * 365 * 5))
        return (start + delta).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    if t == "STRING":
        if faker:
            return _fake_string(prop_name, faker, scale_factor)
    return f"{prop_name}_{idx}"


def generate_invalid_value(prop_type, rng, scale_factor, idx, prop_name, faker=None):
    t = _canonical_prop_type(prop_type)
    if t in {"INT", "FLOAT", "BOOLEAN"}:
        return f"invalid_{prop_name}_{idx}"
    if t == "DATE":
        return f"not-a-date-{idx}"
    if t == "DATETIME":
        return f"not-a-datetime-{idx}"
    if t == "STRING":
        return generate_value("INT", rng, scale_factor, idx, prop_name, faker)
    return None


def _typo_variant(text, rng, forbidden=None):
    forbidden = forbidden or set()
    if not text:
        return "typo"

    for _ in range(8):
        if len(text) >= 2 and rng.random() < 0.5:
            idx = rng.randint(0, len(text) - 2)
            chars = list(text)
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
            candidate = "".join(chars)
        else:
            idx = rng.randint(0, len(text) - 1)
            chars = list(text)
            source = chars[idx]
            if source.isupper():
                alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            elif source.islower():
                alphabet = "abcdefghijklmnopqrstuvwxyz"
            elif source.isdigit():
                alphabet = "0123456789"
            else:
                alphabet = "abcdefghijklmnopqrstuvwxyz"
            replacements = [char for char in alphabet if char != source]
            chars[idx] = rng.choice(replacements) if replacements else "x"
            candidate = "".join(chars)
        if candidate != text and candidate not in forbidden:
            return candidate

    candidate = f"{text}_typo"
    while candidate in forbidden:
        candidate = f"{candidate}x"
    return candidate


def _fresh_label_name(rng, schema_labels, existing, prefix):
    while True:
        candidate = f"{prefix}{rng.randint(1, 999999)}"
        if candidate not in schema_labels and candidate not in existing:
            return candidate


def _fresh_prop_name(rng, existing, prefix):
    while True:
        candidate = f"{prefix}_{rng.randint(1, 999999)}"
        if candidate not in existing:
            return candidate


def _generate_fresh_props(rng, scale_factor, idx, faker, prefix):
    prop_count = 1 + rng.randint(0, 1)
    props = {}
    for extra_idx in range(prop_count):
        key = _fresh_prop_name(rng, set(props.keys()), prefix)
        props[key] = generate_value("STRING", rng, scale_factor, idx + extra_idx, key, faker)
    return props


def _preferred_property_key(record_spec, props=None):
    if props is None:
        required_keys = sorted(record_spec.required.keys())
        if required_keys:
            return required_keys
        return sorted(record_spec.optional.keys())

    required_keys = [key for key in sorted(record_spec.required.keys()) if key in props]
    if required_keys:
        return required_keys
    return [key for key in sorted(record_spec.optional.keys()) if key in props]


def _apply_nonconforming_mutations(
    labels,
    props,
    option,
    mutation_rates,
    rng,
    scale_factor,
    idx,
    schema_labels,
    faker=None,
):
    mutated_labels = list(labels)
    mutated_props = dict(props)
    mutations = []

    if (
        mutation_rates.extra_label_probability
        and not option.label_open
        and rng.random() < mutation_rates.extra_label_probability
    ):
        extra_label = _fresh_label_name(
            rng,
            schema_labels,
            set(mutated_labels),
            "MutatedLabel",
        )
        mutated_labels.append(extra_label)
        mutations.append({"kind": "extra_label", "before": None, "after": extra_label})

    if (
        mutation_rates.extra_property_probability
        and not option.record_spec.open
        and rng.random() < mutation_rates.extra_property_probability
    ):
        extra_key = _fresh_prop_name(rng, set(mutated_props.keys()), "mutated_prop")
        mutated_props[extra_key] = generate_value(
            "STRING",
            rng,
            scale_factor,
            idx,
            extra_key,
            faker,
        )
        mutations.append(
            {
                "kind": "extra_property",
                "property": extra_key,
                "before": None,
                "after": mutated_props[extra_key],
            }
        )

    if (
        mutation_rates.invalid_optional_property_probability
        and option.record_spec.optional
        and rng.random() < mutation_rates.invalid_optional_property_probability
    ):
        bad_key = rng.choice(sorted(option.record_spec.optional.keys()))
        before = mutated_props.get(bad_key)
        mutated_props[bad_key] = generate_invalid_value(
            option.record_spec.optional[bad_key],
            rng,
            scale_factor,
            idx,
            bad_key,
            faker,
        )
        mutations.append(
            {
                "kind": "invalid_optional_property",
                "property": bad_key,
                "expected": option.record_spec.optional[bad_key],
                "before": before,
                "after": mutated_props[bad_key],
            }
        )

    if (
        mutation_rates.wrong_property_datatype_probability
        and (option.record_spec.required or option.record_spec.optional)
        and rng.random() < mutation_rates.wrong_property_datatype_probability
    ):
        candidate_keys = _preferred_property_key(option.record_spec)
        bad_key = rng.choice(candidate_keys)
        prop_type = option.record_spec.required.get(bad_key) or option.record_spec.optional[bad_key]
        before = mutated_props.get(bad_key)
        mutated_props[bad_key] = generate_invalid_value(
            prop_type,
            rng,
            scale_factor,
            idx,
            bad_key,
            faker,
        )
        mutations.append(
            {
                "kind": "wrong_property_datatype",
                "property": bad_key,
                "expected": prop_type,
                "before": before,
                "after": mutated_props[bad_key],
            }
        )

    if (
        mutation_rates.missing_required_property_probability
        and option.record_spec.required
        and rng.random() < mutation_rates.missing_required_property_probability
    ):
        missing_key = rng.choice(sorted(option.record_spec.required.keys()))
        removed_value = mutated_props.pop(missing_key, None)
        mutations.append(
            {
                "kind": "missing_required_property",
                "property": missing_key,
                "before": removed_value,
                "after": None,
            }
        )

    if (
        mutation_rates.missing_required_label_probability
        and option.labels
        and rng.random() < mutation_rates.missing_required_label_probability
    ):
        removable = [label for label in mutated_labels if label in option.labels]
        if removable:
            removed_label = rng.choice(removable)
            mutated_labels.remove(removed_label)
            mutations.append(
                {
                    "kind": "missing_required_label",
                    "before": removed_label,
                    "after": None,
                }
            )

    if (
        mutation_rates.typo_label_probability
        and option.labels
        and rng.random() < mutation_rates.typo_label_probability
    ):
        typo_candidates = [label for label in mutated_labels if label in option.labels]
        if typo_candidates:
            original = rng.choice(typo_candidates)
            typo = _typo_variant(
                original,
                rng,
                forbidden=set(schema_labels) | set(mutated_labels),
            )
            mutated_labels[mutated_labels.index(original)] = typo
            mutations.append({"kind": "typo_label", "before": original, "after": typo})

    if (
        mutation_rates.typo_property_key_probability
        and mutated_props
        and rng.random() < mutation_rates.typo_property_key_probability
    ):
        typo_candidates = _preferred_property_key(option.record_spec, mutated_props)
        if typo_candidates:
            original = rng.choice(typo_candidates)
            value = mutated_props.pop(original)
            typo_key = _typo_variant(
                original,
                rng,
                forbidden=set(mutated_props.keys())
                | set(option.record_spec.required.keys())
                | set(option.record_spec.optional.keys()),
            )
            mutated_props[typo_key] = value
            mutations.append(
                {
                    "kind": "typo_property_key",
                    "property": original,
                    "before": original,
                    "after": typo_key,
                    "value": value,
                }
            )

    return sorted(dict.fromkeys(mutated_labels)), mutated_props, mutations


def _append_fresh_node(nodes, rng, scale_factor, schema_labels, faker=None, mutation_report=None):
    label = _fresh_label_name(rng, schema_labels, set(), "FreshNode")
    props = _generate_fresh_props(rng, scale_factor, len(nodes), faker, "fresh_node_prop")
    node = NodeInstance(f"n{len(nodes)}", [label], props, "__fresh_node__")
    nodes.append(node)
    if mutation_report is not None:
        mutation_report.record_object_mutations(
            node.node_id,
            "node",
            node.type_name,
            [
                {
                    "kind": "fresh_node",
                    "labels": list(node.labels),
                    "properties": dict(node.props),
                }
            ],
        )
    return node


def _append_fresh_edge(edges, nodes, rng, scale_factor, schema_labels, faker=None, mutation_report=None):
    while len(nodes) < 2:
        _append_fresh_node(nodes, rng, scale_factor, schema_labels, faker, mutation_report)
    label = _fresh_label_name(rng, schema_labels, set(), "FreshEdge")
    props = _generate_fresh_props(rng, scale_factor, len(edges), faker, "fresh_edge_prop")
    source = rng.choice(nodes)
    target = rng.choice(nodes)
    edge = EdgeInstance(
        f"e{len(edges)}",
        source.node_id,
        target.node_id,
        [label],
        props,
        "__fresh_edge__",
    )
    edges.append(edge)
    if mutation_report is not None:
        mutation_report.record_object_mutations(
            edge.edge_id,
            "edge",
            edge.type_name,
            [
                {
                    "kind": "fresh_edge",
                    "labels": list(edge.labels),
                    "properties": dict(edge.props),
                }
            ],
            source=edge.source,
            target=edge.target,
        )
    return edge


def generate_instances(
    graph_type,
    node_types,
    edge_types,
    scale_factor,
    rng,
    faker=None,
    force_open_extras=0,
    mutation_plan=None,
    mutation_report=None,
):
    nodes = []
    edges = []
    resolved_node_types = []
    resolved_edge_types = []
    mutation_plan = mutation_plan or MutationPlan()

    for element in graph_type.elements:
        if isinstance(element, NodeType):
            resolved_node_types.append(element)
        elif isinstance(element, EdgeType):
            resolved_edge_types.append(element)
        elif isinstance(element, str):
            if element in node_types:
                resolved_node_types.append(node_types[element])
            elif element in edge_types:
                resolved_edge_types.append(edge_types[element])
            else:
                spec = LabelPropSpec(None, False, [], False)
                resolved_node_types.append(NodeType(element, spec))

    semantics = SchemaSemantics(node_types, edge_types)
    label_pool = set(node_types.keys()) | set(edge_types.keys())
    for nt in node_types.values():
        label_pool |= _collect_label_names(nt.spec.labels_ast) if nt.spec else set()
    for et in edge_types.values():
        if et.spec:
            label_pool |= _collect_label_names(et.spec.labels_ast)
        if et.left_spec:
            label_pool |= _collect_label_names(et.left_spec.labels_ast)
        if et.right_spec:
            label_pool |= _collect_label_names(et.right_spec.labels_ast)

    def _extra_label(existing):
        for _ in range(5):
            candidate = f"ExtraLabel{rng.randint(1, 9999)}"
            if candidate not in existing:
                return candidate
        for candidate in label_pool:
            if candidate not in existing:
                return candidate
        return None

    def _extra_prop(existing, idx):
        for _ in range(5):
            candidate = f"extra_prop_{rng.randint(1, 9999)}"
            if candidate not in existing:
                return candidate
        return f"extra_prop_{idx}"

    def _generate_record(record_spec, idx):
        props = {}
        for key, prop_type in record_spec.required.items():
            props[key] = generate_value(prop_type, rng, scale_factor, idx, key, faker)
        for key, prop_type in record_spec.optional.items():
            if rng.random() < 0.3:
                continue
            props[key] = generate_value(prop_type, rng, scale_factor, idx, key, faker)
        if force_open_extras and record_spec.open:
            for extra_idx in range(force_open_extras):
                extra_key = _extra_prop(set(props.keys()), idx + extra_idx)
                extra_val = generate_value("STRING", rng, scale_factor, idx + extra_idx, extra_key, faker)
                props[extra_key] = extra_val
        return props

    base_nodes_per_type = 10
    nodes_per_type = max(1, int(scale_factor) * base_nodes_per_type)
    for node_type in resolved_node_types:
        if node_type.abstract:
            continue
        mutation_rates = mutation_plan.for_type(node_type.name)
        options = semantics.eval_node_type(node_type.name)
        if not options:
            raise ValueError(f"No valid instances for node type {node_type.name}")
        for _ in range(nodes_per_type):
            option = rng.choice(options)
            labels = sorted(option.labels)
            if force_open_extras and option.label_open:
                for _ in range(force_open_extras):
                    extra = _extra_label(set(labels))
                    if extra:
                        labels.append(extra)
            props = _generate_record(option.record_spec, len(nodes))
            schema_labels = list(labels)
            schema_props = dict(props)
            labels, props, mutations = _apply_nonconforming_mutations(
                labels,
                props,
                option,
                mutation_rates,
                rng,
                scale_factor,
                len(nodes),
                label_pool,
                faker,
            )
            node_id = f"n{len(nodes)}"
            nodes.append(
                NodeInstance(
                    node_id,
                    labels,
                    props,
                    node_type.name,
                    schema_labels=schema_labels,
                    schema_props=schema_props,
                )
            )
            if mutation_report is not None:
                mutation_report.record_object_mutations(
                    node_id,
                    "node",
                    node_type.name,
                    mutations,
                )
            if (
                mutation_rates.fresh_node_probability
                and rng.random() < mutation_rates.fresh_node_probability
            ):
                _append_fresh_node(
                    nodes,
                    rng,
                    scale_factor,
                    label_pool,
                    faker,
                    mutation_report,
                )

    base_edges_per_type = 15
    edges_per_type = max(1, int(scale_factor) * base_edges_per_type)
    for edge_type in resolved_edge_types:
        if edge_type.abstract:
            continue
        mutation_rates = mutation_plan.for_type(edge_type.name)
        options = semantics.eval_edge_type(edge_type.name)
        if not options:
            continue
        option_candidates = []
        for opt in options:
            sources = [
                n
                for n in nodes
                if _labels_conform(set(n.schema_labels), opt.source)
                and _record_conforms(n.schema_props, opt.source.record_spec)
            ]
            targets = [
                n
                for n in nodes
                if _labels_conform(set(n.schema_labels), opt.target)
                and _record_conforms(n.schema_props, opt.target.record_spec)
            ]
            if sources and targets:
                option_candidates.append((opt, sources, targets))
        if not option_candidates:
            continue
        for _ in range(edges_per_type):
            opt, sources, targets = rng.choice(option_candidates)
            source = rng.choice(sources)
            target = rng.choice(targets)
            labels = sorted(opt.edge.labels)
            if not labels:
                if opt.edge.label_open:
                    labels = [edge_type.name]
                else:
                    raise ValueError(
                        f"Edge type {edge_type.name} produces unlabeled edges; cannot map to GraphML"
                    )
            if force_open_extras and opt.edge.label_open:
                for _ in range(force_open_extras):
                    extra = _extra_label(set(labels))
                    if extra:
                        labels.append(extra)
            props = _generate_record(opt.edge.record_spec, len(edges))
            labels, props, mutations = _apply_nonconforming_mutations(
                labels,
                props,
                opt.edge,
                mutation_rates,
                rng,
                scale_factor,
                len(edges),
                label_pool,
                faker,
            )
            edge_id = f"e{len(edges)}"
            edges.append(
                EdgeInstance(edge_id, source.node_id, target.node_id, labels, props, edge_type.name)
            )
            if mutation_report is not None:
                mutation_report.record_object_mutations(
                    edge_id,
                    "edge",
                    edge_type.name,
                    mutations,
                    source=source.node_id,
                    target=target.node_id,
                )
            if (
                mutation_rates.fresh_edge_probability
                and rng.random() < mutation_rates.fresh_edge_probability
            ):
                _append_fresh_edge(
                    edges,
                    nodes,
                    rng,
                    scale_factor,
                    label_pool,
                    faker,
                    mutation_report,
                )
    node_prop_names = set()
    edge_prop_names = set()
    for node in nodes:
        node_prop_names.update(node.props.keys())
    for edge in edges:
        edge_prop_names.update(edge.props.keys())
    return nodes, edges, node_prop_names, edge_prop_names


def build_graphml(nodes, edges, node_prop_names, edge_prop_names):
    edge_has_labels = any(len(edge.labels) > 1 for edge in edges)

    ns = "http://graphml.graphdrawing.org/xmlns"
    xsi = "http://www.w3.org/2001/XMLSchema-instance"
    ET.register_namespace("", ns)
    ET.register_namespace("xsi", xsi)
    root = ET.Element(
        f"{{{ns}}}graphml",
        {f"{{{xsi}}}schemaLocation": f"{ns} {ns}/1.0/graphml.xsd"},
    )

    key_ids = {}
    key_ids[("node", "labels")] = "labels"
    edge_label_key = "label" if "label" not in node_prop_names else "e_label"
    key_ids[("edge", "label")] = edge_label_key
    if edge_has_labels:
        key_ids[("edge", "labels")] = "labels" if ("node", "labels") not in key_ids else "e_labels"

    for name in sorted(node_prop_names):
        key_ids[("node", name)] = name
    for name in sorted(edge_prop_names):
        if name in node_prop_names or name == "labels":
            key_ids[("edge", name)] = f"e_{name}"
        else:
            key_ids[("edge", name)] = name

    for name in ["labels"] + sorted(node_prop_names):
        ET.SubElement(
            root,
            f"{{{ns}}}key",
            {"id": key_ids[("node", name)], "for": "node", "attr.name": name},
        )
    ET.SubElement(
        root,
        f"{{{ns}}}key",
        {"id": key_ids[("edge", "label")], "for": "edge", "attr.name": "label"},
    )
    if edge_has_labels:
        ET.SubElement(
            root,
            f"{{{ns}}}key",
            {"id": key_ids[("edge", "labels")], "for": "edge", "attr.name": "labels"},
        )
    for name in sorted(edge_prop_names):
        ET.SubElement(
            root,
            f"{{{ns}}}key",
            {"id": key_ids[("edge", name)], "for": "edge", "attr.name": name},
        )

    graph = ET.SubElement(root, f"{{{ns}}}graph", {"id": "G", "edgedefault": "directed"})

    for node in nodes:
        label_str = ":" + ":".join(node.labels) if node.labels else ""
        node_el = ET.SubElement(graph, f"{{{ns}}}node", {"id": node.node_id, "labels": label_str})
        data_labels = ET.SubElement(node_el, f"{{{ns}}}data", {"key": key_ids[("node", "labels")]})
        data_labels.text = label_str
        for key, value in node.props.items():
            data_el = ET.SubElement(node_el, f"{{{ns}}}data", {"key": key_ids[("node", key)]})
            data_el.text = str(value)

    for edge in edges:
        label_str = ":" + ":".join(edge.labels) if edge.labels else ""
        primary_label = edge.labels[0] if edge.labels else ""
        edge_el = ET.SubElement(
            graph,
            f"{{{ns}}}edge",
            {"id": edge.edge_id, "source": edge.source, "target": edge.target, "label": primary_label},
        )
        data_label = ET.SubElement(edge_el, f"{{{ns}}}data", {"key": key_ids[("edge", "label")]})
        data_label.text = primary_label
        if edge_has_labels:
            data_labels = ET.SubElement(edge_el, f"{{{ns}}}data", {"key": key_ids[("edge", "labels")]})
            data_labels.text = label_str
        for key, value in edge.props.items():
            data_el = ET.SubElement(edge_el, f"{{{ns}}}data", {"key": key_ids[("edge", key)]})
            data_el.text = str(value)

    return ET.ElementTree(root)


def _infer_graphml_type(values):
    has_string = any(isinstance(value, str) for value in values)
    has_bool = any(isinstance(value, bool) for value in values)
    has_float = any(isinstance(value, float) for value in values)
    has_int = any(isinstance(value, int) and not isinstance(value, bool) for value in values)
    if has_string:
        return "string"
    if has_bool and not (has_int or has_float):
        return "boolean"
    if has_float:
        return "double"
    if has_int:
        return "int"
    if has_bool:
        return "boolean"
    return "string"


def build_graphml_tinkerpop(nodes, edges, include_labels_prop=False):
    ns = "http://graphml.graphdrawing.org/xmlns"
    xsi = "http://www.w3.org/2001/XMLSchema-instance"
    ET.register_namespace("", ns)
    ET.register_namespace("xsi", xsi)
    root = ET.Element(
        f"{{{ns}}}graphml",
        {f"{{{xsi}}}schemaLocation": f"{ns} {ns}/1.0/graphml.xsd"},
    )

    node_values = {}
    edge_values = {}
    labels_prop_node = None
    labels_prop_edge = None
    if include_labels_prop:
        labels_prop_node = "labels" if "labels" not in node_values else "tp_labels"
        labels_prop_edge = "labels" if "labels" not in edge_values else "tp_labels"

    for node in nodes:
        for key, value in node.props.items():
            node_values.setdefault(key, []).append(value)
        if include_labels_prop and len(node.labels) > 1:
            node_values.setdefault(labels_prop_node, []).append(":".join(node.labels))
    for edge in edges:
        for key, value in edge.props.items():
            edge_values.setdefault(key, []).append(value)
        if include_labels_prop and len(edge.labels) > 1:
            edge_values.setdefault(labels_prop_edge, []).append(":".join(edge.labels))

    ET.SubElement(
        root,
        f"{{{ns}}}key",
        {"id": "labelV", "for": "node", "attr.name": "labelV", "attr.type": "string"},
    )
    ET.SubElement(
        root,
        f"{{{ns}}}key",
        {"id": "labelE", "for": "edge", "attr.name": "labelE", "attr.type": "string"},
    )

    for name, values in sorted(node_values.items()):
        ET.SubElement(
            root,
            f"{{{ns}}}key",
            {"id": name, "for": "node", "attr.name": name, "attr.type": _infer_graphml_type(values)},
        )
    for name, values in sorted(edge_values.items()):
        ET.SubElement(
            root,
            f"{{{ns}}}key",
            {"id": name, "for": "edge", "attr.name": name, "attr.type": _infer_graphml_type(values)},
        )

    graph = ET.SubElement(root, f"{{{ns}}}graph", {"id": "G", "edgedefault": "directed"})

    for node in nodes:
        node_el = ET.SubElement(graph, f"{{{ns}}}node", {"id": node.node_id})
        label = node.labels[0] if node.labels else "vertex"
        data_label = ET.SubElement(node_el, f"{{{ns}}}data", {"key": "labelV"})
        data_label.text = label
        for key, value in node.props.items():
            data_el = ET.SubElement(node_el, f"{{{ns}}}data", {"key": key})
            data_el.text = str(value)
        if include_labels_prop and len(node.labels) > 1:
            data_labels = ET.SubElement(node_el, f"{{{ns}}}data", {"key": labels_prop_node})
            data_labels.text = ":".join(node.labels)

    for edge in edges:
        edge_el = ET.SubElement(
            graph,
            f"{{{ns}}}edge",
            {"id": edge.edge_id, "source": edge.source, "target": edge.target},
        )
        label = edge.labels[0] if edge.labels else "edge"
        data_label = ET.SubElement(edge_el, f"{{{ns}}}data", {"key": "labelE"})
        data_label.text = label
        for key, value in edge.props.items():
            data_el = ET.SubElement(edge_el, f"{{{ns}}}data", {"key": key})
            data_el.text = str(value)
        if include_labels_prop and len(edge.labels) > 1:
            data_labels = ET.SubElement(edge_el, f"{{{ns}}}data", {"key": labels_prop_edge})
            data_labels.text = ":".join(edge.labels)

    return ET.ElementTree(root)


def write_graphson3_no_types(nodes, edges, path, wrap=False):
    out_edges = {}
    in_edges = {}
    for edge in edges:
        label = edge.labels[0] if edge.labels else "edge"
        out_edges.setdefault(edge.source, {}).setdefault(label, []).append(
            {"id": edge.edge_id, "inV": edge.target, "properties": edge.props}
        )
        in_edges.setdefault(edge.target, {}).setdefault(label, []).append(
            {"id": edge.edge_id, "outV": edge.source, "properties": edge.props}
        )

    vp_id = 0
    vertices = []
    for node in nodes:
        vertex = {
            "id": node.node_id,
            "label": node.labels[0] if node.labels else "vertex",
        }
        if node.props:
            props = {}
            for key, value in node.props.items():
                props[key] = [{"id": vp_id, "value": value}]
                vp_id += 1
            vertex["properties"] = props
        if node.node_id in out_edges:
            vertex["outE"] = out_edges[node.node_id]
        if node.node_id in in_edges:
            vertex["inE"] = in_edges[node.node_id]
        vertices.append(vertex)

    with open(path, "w", encoding="utf-8") as handle:
        if wrap:
            handle.write(json.dumps({"vertices": vertices}, ensure_ascii=False))
            handle.write("\n")
        else:
            for vertex in vertices:
                handle.write(json.dumps(vertex, ensure_ascii=False))
                handle.write("\n")


def write_oracle_graphson(nodes, edges, path, mode="NORMAL"):
    vertices = []
    for node in nodes:
        vertex = {}
        vertex.update(node.props)
        vertex["_id"] = node.node_id
        vertex["_type"] = "vertex"
        vertices.append(vertex)

    edge_items = []
    for edge in edges:
        edge_item = {}
        edge_item.update(edge.props)
        edge_item["_id"] = edge.edge_id
        edge_item["_type"] = "edge"
        edge_item["_outV"] = edge.source
        edge_item["_inV"] = edge.target
        edge_item["_label"] = edge.labels[0] if edge.labels else "edge"
        edge_items.append(edge_item)

    payload = {"graph": {"mode": mode, "vertices": vertices, "edges": edge_items}}
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False))
        handle.write("\n")


def main(argv):
    parser = argparse.ArgumentParser(description="Generate GraphML data from PG-Schema")
    parser.add_argument("schema", help="Path to PG-Schema file")
    parser.add_argument("scale", type=int, help="Scale factor (controls output size)")
    parser.add_argument("-o", "--out", help="Output GraphML file path")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("-f", "--fake", action="store_true", help="Use faker for more realistic values")
    parser.add_argument(
        "--format",
        choices=["apoc", "graphml-tp", "graphson3", "oracle-graphson"],
        default="apoc",
        help="Output format: apoc (Neo4j GraphML), graphml-tp (TinkerPop GraphML), graphson3 (GraphSON 3.0 without embedded types), oracle-graphson (Oracle Property Graph GraphSON)",
    )
    parser.add_argument(
        "--tp-labels-prop",
        action="store_true",
        help="For graphml-tp, include full label set as a property when multiple labels exist",
    )
    parser.add_argument(
        "--graphson-wrap",
        action="store_true",
        help="For graphson3, output a single JSON object with a vertices array instead of line-delimited",
    )
    parser.add_argument(
        "--open-extra",
        action="store_true",
        help="When OPEN is used, force extra labels/properties beyond the schema",
    )
    parser.add_argument(
        "--open-extra-count",
        type=int,
        default=1,
        help="How many extra labels/properties to add per OPEN element (requires --open-extra)",
    )
    parser.add_argument(
        "--mutation-config",
        help="Optional JSON file with non-conforming mutation probabilities, optionally per type",
    )
    parser.add_argument(
        "--mutation-report",
        help="Optional JSON file path for mutation statistics and per-object mutation details",
    )
    parser.add_argument(
        "--mutation-fresh-node-prob",
        type=float,
        default=None,
        help="Probability of adding a fresh unrelated node after generating a node of a type",
    )
    parser.add_argument(
        "--mutation-fresh-edge-prob",
        type=float,
        default=None,
        help="Probability of adding a fresh unrelated edge after generating an edge of a type",
    )
    parser.add_argument(
        "--mutation-extra-label-prob",
        type=float,
        default=None,
        help="Probability of adding a fresh label to a closed node/edge",
    )
    parser.add_argument(
        "--mutation-extra-prop-prob",
        type=float,
        default=None,
        help="Probability of adding a fresh property to a closed node/edge record",
    )
    parser.add_argument(
        "--mutation-bad-optional-prop-prob",
        type=float,
        default=None,
        help="Probability of assigning an invalid value to an OPTIONAL property",
    )
    parser.add_argument(
        "--mutation-wrong-prop-datatype-prob",
        type=float,
        default=None,
        help="Probability of assigning an invalid value to a schema-defined property",
    )
    parser.add_argument(
        "--mutation-missing-required-prop-prob",
        type=float,
        default=None,
        help="Probability of omitting a required property",
    )
    parser.add_argument(
        "--mutation-missing-required-label-prob",
        type=float,
        default=None,
        help="Probability of omitting a required label",
    )
    parser.add_argument(
        "--mutation-typo-label-prob",
        type=float,
        default=None,
        help="Probability of mutating a required label with a typo",
    )
    parser.add_argument(
        "--mutation-typo-prop-key-prob",
        type=float,
        default=None,
        help="Probability of mutating a schema property key with a typo",
    )
    args = parser.parse_args(argv)

    if args.scale < 1:
        raise SystemExit("Scale factor must be >= 1")
    if args.open_extra and args.open_extra_count < 1:
        raise SystemExit("--open-extra-count must be >= 1")

    with open(args.schema, "r", encoding="utf-8") as f:
        schema_text = f.read()

    node_types, edge_types, graph_types = parse_schema(schema_text)
    if not graph_types:
        raise SystemExit("No GRAPH TYPE found in schema")

    graph_type = graph_types[0]
    rng = random.Random(args.seed)
    faker = None
    if args.fake:
        try:
            from faker import Faker
        except ImportError as exc:
            raise SystemExit(
                "Faker not installed. Install with: python3 -m pip install faker"
            ) from exc
        faker = Faker()
        if args.seed is not None:
            faker.seed_instance(args.seed)

    cli_mutations = {
        "fresh_node_probability": args.mutation_fresh_node_prob,
        "fresh_edge_probability": args.mutation_fresh_edge_prob,
        "extra_label_probability": args.mutation_extra_label_prob,
        "extra_property_probability": args.mutation_extra_prop_prob,
        "invalid_optional_property_probability": args.mutation_bad_optional_prop_prob,
        "wrong_property_datatype_probability": args.mutation_wrong_prop_datatype_prob,
        "missing_required_property_probability": args.mutation_missing_required_prop_prob,
        "missing_required_label_probability": args.mutation_missing_required_label_prob,
        "typo_label_probability": args.mutation_typo_label_prob,
        "typo_property_key_probability": args.mutation_typo_prop_key_prob,
    }
    try:
        mutation_plan = load_mutation_plan(args.mutation_config, cli_mutations)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid mutation configuration: {exc}") from exc

    open_extra_count = args.open_extra_count if args.open_extra else 0
    mutation_report = MutationReport() if args.mutation_report else None
    nodes, edges, node_prop_names, edge_prop_names = generate_instances(
        graph_type,
        node_types,
        edge_types,
        args.scale,
        rng,
        faker,
        force_open_extras=open_extra_count,
        mutation_plan=mutation_plan,
        mutation_report=mutation_report,
    )

    out_path = args.out
    if not out_path:
        base = args.schema.rsplit("/", 1)[-1]
        if "." in base:
            base = base.rsplit(".", 1)[0]
        if args.format == "graphson3":
            out_path = f"{base}-generated.graphson"
        elif args.format == "oracle-graphson":
            out_path = f"{base}-generated.oracle.graphson"
        elif args.format == "graphml-tp":
            out_path = f"{base}-generated.tinkerpop.graphml"
        else:
            out_path = f"{base}-generated.graphml"

    if args.format == "graphson3":
        write_graphson3_no_types(nodes, edges, out_path, wrap=args.graphson_wrap)
    elif args.format == "oracle-graphson":
        write_oracle_graphson(nodes, edges, out_path)
    elif args.format == "graphml-tp":
        tree = build_graphml_tinkerpop(nodes, edges, include_labels_prop=args.tp_labels_prop)
        tree.write(out_path, encoding="utf-8", xml_declaration=True)
    else:
        tree = build_graphml(nodes, edges, node_prop_names, edge_prop_names)
        tree.write(out_path, encoding="utf-8", xml_declaration=True)

    if mutation_report is not None:
        mutation_report.write_json(
            args.mutation_report,
            nodes,
            edges,
            meta={
                "schema": args.schema,
                "graph_type": graph_type.name,
                "seed": args.seed,
                "scale": args.scale,
                "format": args.format,
            },
        )

    print(f"Wrote {out_path} (nodes: {len(nodes)}, edges: {len(edges)})")


if __name__ == "__main__":
    main(sys.argv[1:])
