# inspect_db.py

import chromadb
import pprint # Python's built-in "pretty-print" library
from core.config import load_config # Assuming you have a config loader

# --- SCRIPT CONFIGURATION ---
# How many records to fetch and display.
# Since each segment has 2 entries (text/visual), a limit of 10 will show 5 full segments.
FETCH_LIMIT = 10 

def inspect_collection():
    """
    Connects to ChromaDB, fetches a sample of records, and prints them.
    """
    print("--- ChromaDB Inspection Tool ---")
    
    # 1. Load configuration to get database details
    try:
        config = load_config("config.yaml")
        db_config = config['database']
    except Exception as e:
        print(f"Error loading config.yaml: {e}")
        print("Please ensure config.yaml exists and is correctly formatted.")
        return

    # 2. Connect to the ChromaDB client
    print(f"Connecting to ChromaDB at {db_config['host']}:{db_config['port']}...")
    try:
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
    print(f"\nFetching the first {FETCH_LIMIT} items...")
    
    # Use collection.get() to retrieve records.
    # We include 'metadatas' because that's where all our human-readable info is.
    # We EXCLUDE 'embeddings' because they are just giant arrays of numbers.
    results = collection.get(
        limit=FETCH_LIMIT,
        include=["metadatas"] 
    )
    
    # 5. Print the results in a nicely formatted way
    print("\n--- Sample Records ---")
    pprint.pprint(results)


if __name__ == '__main__':
    inspect_collection()
