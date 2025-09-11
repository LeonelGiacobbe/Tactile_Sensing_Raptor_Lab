import matplotlib.pyplot as plt
import numpy as np

batch_size = [1, 2, 4, 8, 16, 32, 64, 128]
rt_increase = [136.56,  4.05, 21.39, 40.78, 32.90, 49.88, 40.56, 43.46]

plt.figure(figsize=(15,9))

plt.plot(batch_size, rt_increase, color='r', linestyle="solid",
         label="Runtime increase across batch sizes")

# Use log-log scaling
plt.xscale('log', base=2)     # log2 scale for batch sizes

# Ticks: explicitly set x-ticks to your batch_size values
plt.xticks(batch_size, batch_size, fontsize=25)
plt.yticks(fontsize=25)

plt.xlabel('Batch size', fontsize=35)
plt.ylabel('Percentage (%)', fontsize=35)

# Show grid for both major and minor ticks
plt.grid(True, which="both", ls="--", linewidth=0.7)
plt.legend(fontsize=25)

plt.savefig("batch_plot.pdf", bbox_inches="tight")
