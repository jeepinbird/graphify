"""Tests for the Go extractor."""
from pathlib import Path
from graphify.extract import extract_go, extract

FIXTURES = Path(__file__).parent / "fixtures"


def test_go_no_error():
    r = extract_go(FIXTURES / "sample.go")
    assert "error" not in r
    assert r["nodes"]


def test_go_finds_struct_and_funcs():
    r = extract_go(FIXTURES / "sample.go")
    labels = {n["label"] for n in r["nodes"]}
    assert "Server" in labels
    assert any(l.startswith("NewServer") for l in labels)


def test_go_receiver_methods_share_type_node():
    """Methods on the same receiver type must share one canonical type node."""
    r = extract_go(FIXTURES / "sample.go")
    server_nodes = [n for n in r["nodes"] if n["label"] == "Server"]
    # Both Start() and Stop() are on *Server — should produce exactly one Server node
    assert len(server_nodes) == 1


def test_go_receiver_uses_pkg_scope():
    """Type node id should be scoped to directory, not file stem."""
    r = extract_go(FIXTURES / "sample.go")
    server_nodes = [n for n in r["nodes"] if n["label"] == "Server"]
    assert server_nodes
    # Should NOT contain the file stem "sample" in the type node id
    assert "sample" not in server_nodes[0]["id"].split(":")[0]


def test_go_no_dangling_edges():
    r = extract_go(FIXTURES / "sample.go")
    node_ids = {n["id"] for n in r["nodes"]}
    for edge in r["edges"]:
        if edge["relation"] in {"contains", "method", "calls"}:
            assert edge["source"] in node_ids
            assert edge["target"] in node_ids


def test_go_routes_through_extract():
    r = extract([FIXTURES / "sample.go"])
    assert any(n["label"] == "Server" for n in r["nodes"])
