import json

cov_path = 'out/cov-gencog-complete/cov.json'

comps = ['relay', 'topi', 'tir', 'te', 'arith', 'target', 'runtime']

# Load coverage report
with open(cov_path, 'r') as f:
    cov = json.load(f)

# Compute component statistics
comp_stat = {c: 0 for c in comps}
for file in cov['files']:
    line_cov = file['line_covered']
    path_split = file['filename'].split('/')
    for c in comps:
        if c in path_split:
            comp_stat[c] += line_cov
            break

print(comp_stat)
