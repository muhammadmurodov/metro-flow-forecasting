import networkx as nx

from src.network_view import marker_sizes, separated_layout, no_collisions


def test_demand_changes_marker_size_and_focused_station_stays_clear():
    graph = nx.path_graph(25)
    forecast = {i: {"value": i * 100} for i in graph.nodes}
    sizes = marker_sizes(forecast, selected=12)
    assert sizes[24] > sizes[0]
    layout = separated_layout(graph, sizes, selected=12)
    assert no_collisions(layout, sizes, selected=12)


def test_dense_connected_network_has_clear_markers():
    graph = nx.watts_strogatz_graph(81, 4, .17, seed=8)
    forecast = {i: {"value": 25 + (i * 47) % 900} for i in graph.nodes}
    sizes = marker_sizes(forecast, selected=30)
    layout = separated_layout(graph, sizes, selected=30)
    assert no_collisions(layout, sizes, selected=30)
