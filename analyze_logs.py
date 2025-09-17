import os
import re
from statistics import mean

# Directory containing the logs
LOG_DIR = 'logs'

# Regex patterns
final_solution_pattern = re.compile(r'Final Solution: cost=\[([-\d\.]+)\], size=\[\d+\], elements=\{.*\}')
time_pattern = re.compile(r'--- Finished in ([\d\.]+) seconds ---')
instance_size_pattern = re.compile(r'(\d+)')

# Data structure: {instance_size: [(cost, time), ...]}
results = {}

for filename in os.listdir(LOG_DIR):
    if not filename.endswith('.log'):
        continue
    # Extract instance size from filename (first number found)
    match = instance_size_pattern.search(filename)
    if not match:
        continue
    instance_size = int(match.group(1))
    with open(os.path.join(LOG_DIR, filename), 'r', encoding='utf-8') as f:
        lines = f.readlines()
    cost = None
    time = None
    for line in lines:
        m_cost = final_solution_pattern.search(line)
        if m_cost:
            cost = float(m_cost.group(1))
        m_time = time_pattern.search(line)
        if m_time:
            time = float(m_time.group(1))
    if cost is not None and time is not None:
        results.setdefault(instance_size, []).append((cost, time))

# Print summary
print(f"{'Instance Size':>15} | {'Best Cost':>10} | {'Avg Cost':>10} | {'Avg Time (s)':>12}")
print('-'*55)
for size in sorted(results):
    costs = [c for c, t in results[size]]
    times = [t for c, t in results[size]]
    best_cost = max(costs)
    avg_cost = mean(costs)
    avg_time = mean(times)
    print(f"{size:15} | {best_cost:10.2f} | {avg_cost:10.2f} | {avg_time:12.2f}")
