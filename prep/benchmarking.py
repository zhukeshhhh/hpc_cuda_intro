import csv
import matplotlib.pyplot as plt

csv_file = 'benchmark.csv'

# Чтение данных
Ns, naive_times, tiled_times = [], [], []
with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        N = int(row['N'])
        naive = float(row['naive_ms'])
        tiled = float(row['tiled_ms'])
        Ns.append(N)
        naive_times.append(naive)
        tiled_times.append(tiled)

# ==== ГРАФИК ВРЕМЕНИ ====
plt.figure(figsize=(8, 5))
plt.plot(Ns, naive_times, label='Naive')
plt.plot(Ns, tiled_times, label='Tiled 32x32')
plt.xlabel('Matrix dimension N')
plt.ylabel('Execution time (ms)')
plt.title('CUDA Matrix Multiplication: Naive vs Tiled Shared Memory\n(averaged over multiple runs)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('matrix_mul_benchmark_32.png', dpi=300)
plt.show()

# ==== ГРАФИК УСКОРЕНИЯ ====
speedups = [naive / tiled if tiled > 0 else 0 for naive, tiled in zip(naive_times, tiled_times)]

plt.figure(figsize=(8, 5))
plt.plot(Ns, speedups, marker='o', color='green')
plt.xlabel('Matrix dimension N')
plt.ylabel('Speedup (naive / tiled)')
plt.title('Speedup of Tiled Matrix Multiplication over Naive CUDA\n(averaged over multiple runs)')
plt.grid(True)
plt.tight_layout()
plt.savefig('matrix_mul_speedup.png', dpi=300)
plt.show()
