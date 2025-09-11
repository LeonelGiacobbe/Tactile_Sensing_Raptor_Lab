import matplotlib.pyplot as plt
import numpy as np

batch_size = [1, 2, 4, 8, 16, 32, 64, 128]
ma_runtime = [0.025232, 0.012691, 0.019566, 0.036248, 0.068352, 0.169282, 0.330727, 0.690623]
sa_runtime = [0.010666, 0.012198, 0.016118, 0.025747, 0.051429, 0.112945, 0.235298, 0.481402]

plt.figure(figsize=(15,9))

plt.plot(batch_size, ma_runtime, color='g', linestyle="solid",
         label="Multi-agent model runtime across batch sizes")
plt.plot(batch_size, sa_runtime, color='r', linestyle="solid",
         label="Single-agent model runtime across batch sizes")

# Use log-log scaling
plt.xscale('log', base=2)     # log2 scale for batch sizes
plt.yscale('log')             # log scale for runtimes

# Ticks: explicitly set x-ticks to your batch_size values
plt.xticks(batch_size, batch_size, fontsize=25)
plt.yticks(fontsize=25)

plt.xlabel('Batch size', fontsize=35)
plt.ylabel('Time (s)', fontsize=35)

# Show grid for both major and minor ticks
plt.grid(True, which="both", ls="--", linewidth=0.7)
plt.legend(fontsize=25)

plt.savefig("runtime_plot.pdf", bbox_inches="tight")
