import csv
import matplotlib.pyplot as plt

# read data
Ns, naive_times, tiled_times = [], [], []
with open('benchmark.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        Ns.append(int(row['N']))
        naive_times.append(float(row['naive_ms']))
        tiled_times.append(float(row['tiled_ms']))

# plot
plt.figure()
plt.plot(Ns, naive_times, label='Naive')
plt.plot(Ns, tiled_times, label='Tiled 32x32')
plt.xlabel('Matrix dimension N')
plt.ylabel('Execution time (ms)')
plt.title('CUDA Matrix Multiplication: Naive vs Tiled Shared Memory')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('matrix_mul_benchmark_32_8192-16384.png', dpi=300)
plt.show()
