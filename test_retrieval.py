from rag import retrieve

while True:
    q = input("\nQuestion (or 'quit'): ")
    if q.lower() == "quit":
        break
    for doc, meta, dist in retrieve(q):
        print(f"\n--- {meta['source']} p.{meta['page']} (distance {dist:.3f}) ---")
        print(doc[:300])