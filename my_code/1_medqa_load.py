import json

def load_medqa_jsonl(path, limit=3):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            example = json.loads(line)
            data.append(example)
    return data

path = "/Users/UChicago/classes/CMSC_25700/Project/data_clean/questions/US/test.jsonl"  # <-- change this

examples = load_medqa_jsonl(path, limit=3)

print("Number of examples loaded:", len(examples))
print("Keys in one example:", examples[0].keys())
print("First example:", examples[0])
