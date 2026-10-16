from src.dataloader.load_data import load_all_data
from src.warehouse.graph_builder import build_warehouse_graph


def main():
    data = load_all_data()

    support_points = data["support_points"]

    graph = build_warehouse_graph(support_points)

    print("=== WAREHOUSE GRAPH ===")
    print(f"Total nodes: {graph.number_of_nodes()}")
    print(f"Total edges: {graph.number_of_edges()}")

    print("\n=== EDGES ===")

    for start, end, attrs in graph.edges(data=True):
        print(
            f"{start} <-> {end} | "
            f"distance={attrs['weight']:.2f} | "
            f"type={attrs['edge_type']}"
        )


if __name__ == "__main__":
    main()
