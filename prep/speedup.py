import csv
import matplotlib.pyplot as plt

# Путь к файлу с результатами
csv_file = 'benchmark.csv'

# Списки для хранения данных
Ns = []
speedups = []

# Считываем данные и вычисляем ускорение
with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        N = int(row['N'])
        naive = float(row['naive_ms'])
        tiled = float(row['tiled_ms'])

        # Защита от деления на ноль
        if tiled > 0:
            speedup = naive / tiled
            Ns.append(N)
            speedups.append(speedup)

# Строим график ускорения
plt.figure()
plt.plot(Ns, speedups, color='green')
plt.xlabel('Matrix dimension N')
plt.ylabel('Speedup (naive / tiled)')
plt.title('Speedup of Tiled Matrix Multiplication over Naive CUDA')
plt.grid(True)
plt.tight_layout()
plt.savefig('matrix_mul_speedup.png', dpi=300)
plt.show()
