import json

base_path = 'out/cov-muffin-dag/cov.json'
diff_path = 'out/cov-typefuzz-muffin/cov.json'

with open(base_path, 'r') as f:
    base = json.load(f)
with open(diff_path, 'r') as f:
    diff = json.load(f)
result = [(bf['filename'], df['line_covered'] - bf['line_covered']) for bf, df in
          zip(base['files'], diff['files'])]
result.sort(key=lambda p: abs(p[1]), reverse=True)
for p, d in result:
    if d != 0:
        print(p, d)
