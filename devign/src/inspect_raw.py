import json

with open('data/raw/dataset.json', 'r') as f:
    data = json.load(f)

sample = data[0]
print(sample.keys())

print(sample['target'])
print(sample['func'])
print(f"Total samples: {len(data)}")
print(f"Vulnerable :{sum(d['target'] == 1 for d in data)}")