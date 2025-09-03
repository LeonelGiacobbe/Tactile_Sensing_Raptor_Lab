import matplotlib.pyplot as plt
import csv
import numpy as np

gripper_pos = []
timestamps = []

with open('pd_position_log.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        gripper_pos.append(float(row[0]))
        timestamps.append(float(row[1]))

plt.figure(figsize=(15,9))

# Plot all points
plt.plot(timestamps, gripper_pos, color='g', linestyle="solid", linewidth = 2.,
         label="Gripper position across time")

# Y-axis: steps of 5, with ±5 buffer
y_min = min(gripper_pos)- 5
y_max = max(gripper_pos) + 5
plt.ylim(y_min, y_max)
plt.yticks(np.arange(y_min, y_max + 0.1, 5), fontsize=25)  # 0.1 to include top edge

# X-axis: keep all points, set ticks every 10
x_min = min(timestamps)
x_max = max(timestamps)
plt.xlim(x_min, x_max)
plt.xticks(np.arange(x_min, x_max + 0.1, 10), fontsize=25)  # tick interval only

plt.xlabel('Time (s)', fontsize=25)
plt.ylabel('Gripper opening in mm', fontsize=25)
plt.title('PD gripper opening graph', fontsize=25)
plt.grid()
plt.legend(fontsize=25)
plt.savefig("pd_position_plot.pdf")

