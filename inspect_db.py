import argparse
import pprint

# --- SCRIPT CONFIGURATION ---
# How many records to fetch and display.
# Since each segment has 2 entries (text/visual), a limit of 10 will show 5 full segments.
FETCH_LIMIT = 10


def _normalize_fetch_limit(fetch_limit: int) -> int:
    if type(fetch_limit) is not int or fetch_limit <= 0:
        raise ValueError("fetch_limit must be a positive integer")
    return fetch_limit


def _positive_fetch_limit(raw_value: str) -> int:
    try:
        return _normalize_fetch_limit(int(raw_value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc


def _load_config():
    from core.config import load_config

    return load_config()


def inspect_collection(fetch_limit: int = FETCH_LIMIT) -> None:
    """
    Connects to ChromaDB, fetches a sample of records, and prints them.
    """
    print("--- ChromaDB Inspection Tool ---")
    try:
        fetch_limit = _normalize_fetch_limit(fetch_limit)
    except ValueError as e:
        print(f"Invalid fetch limit: {e}")
        return

    # 1. Load configuration to get database details
    try:
        config = _load_config()
        db_config = config['database']
    except Exception as e:
        print(f"Error loading config: {e}")
        print("Please ensure CONFIG_PATH or config.yaml is correctly formatted.")
        return

    # 2. Connect to the ChromaDB client
    print(f"Connecting to ChromaDB at {db_config['host']}:{db_config['port']}...")
    try:
        import chromadb

        client = chromadb.HttpClient(host=db_config['host'], port=db_config['port'])
        collection = client.get_collection(name=db_config['collection_name'])
        print(f"Successfully connected to collection '{db_config['collection_name']}'.")
    except Exception as e:
        print(f"Failed to connect to ChromaDB. Is the Docker container running? Error: {e}")
        return

    # 3. Get the total number of items in the collection
    total_items = collection.count()
    print(f"\nCollection contains {total_items} total items (vectors).")

    if total_items == 0:
        print("The collection is empty.")
        return

    # 4. Fetch a sample of items from the collection
    print(f"\nFetching the first {fetch_limit} items...")

    # Use collection.get() to retrieve records.
    # We include 'metadatas' because that's where all our human-readable info is.
    # We EXCLUDE 'embeddings' because they are just giant arrays of numbers.
    results = collection.get(
        limit=fetch_limit,
        include=["metadatas"]
    )

    # 5. Print the results in a nicely formatted way
    print("\n--- Sample Records ---")
    pprint.pprint(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a ChromaDB collection sample.")
    parser.add_argument(
        "--limit",
        type=_positive_fetch_limit,
        default=FETCH_LIMIT,
        help="Number of vectors to fetch.",
    )
    args = parser.parse_args()
    inspect_collection(fetch_limit=args.limit)


if __name__ == '__main__':
    main()
