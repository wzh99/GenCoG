import json

base_path = 'out/cov-muffin-dag/cov.json'
diff_path = 'out/cov-typefuzz-complete/cov.json'

comps = ['relay', 'topi', 'tir', 'arith', 'target']

# Load coverage report
with open(base_path, 'r') as f:
    base = json.load(f)
with open(diff_path, 'r') as f:
    diff = json.load(f)

# Compute coverage difference
result = [(bf['filename'], df['line_covered'] - bf['line_covered']) for bf, df in
          zip(base['files'], diff['files'])]
result.sort(key=lambda p: abs(p[1]), reverse=True)

# Compute component statistics
comp_stat = {c: 0 for c in comps}
for p, d in result:
    path_split = p.split('/')
    for c in comps:
        if c in path_split:
            comp_stat[c] += d
            break
    if d != 0:
        print(p, d)

print(sorted(comp_stat.items(), key=lambda p: p[1], reverse=True))
