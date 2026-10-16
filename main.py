from src.dataloader.load_data import load_all_data
def main():
    data = load_all_data()

    positions = data["positions"]
    support_points = data["support_points"]

    print("\n=== STORAGE LOCATIONS ===")
    print(positions.head())
    print(f"Total locations: {len(positions)}")

    print("\n=== SUPPORT POINTS ===")
    print(support_points.head())
    print(f"Total support points: {len(support_points)}")


if __name__ == "__main__":
    main()
