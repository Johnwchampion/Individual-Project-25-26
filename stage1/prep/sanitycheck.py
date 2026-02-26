import json

DATA_PATH = "/scratch/sc23jc3/squad_prepared/squad_chat_formatted.jsonl"

def show_first_n_pairs(path, n_pairs=3):
    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(next(f)) for _ in range(n_pairs * 2)]

    for i in range(n_pairs):
        ctx = records[2*i]
        base = records[2*i + 1]

        print("=" * 100)
        print(f"PAIR {i+1}")
        print("-" * 100)

        print("\n[WITH CONTEXT]")
        print("ID:", ctx["id"])
        print("Condition:", ctx["condition"])
        print("\nSystem:")
        print(ctx["messages"][0]["content"])
        print("\nUser:")
        print(ctx["messages"][1]["content"])

        print("\n" + "-" * 100)

        print("\n[NO CONTEXT]")
        print("ID:", base["id"])
        print("Condition:", base["condition"])
        print("\nSystem:")
        print(base["messages"][0]["content"])
        print("\nUser:")
        print(base["messages"][1]["content"])

        print("\n" + "=" * 100 + "\n")

show_first_n_pairs(DATA_PATH, n_pairs=3)
