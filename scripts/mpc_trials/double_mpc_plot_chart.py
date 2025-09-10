import matplotlib.pyplot as plt
import csv
import numpy as np

gripper_1_pos = []
gripper_2_pos = []
timestamps = []

with open('corn_50mm_double_mpc_position_log.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        gripper_1_pos.append(float(row[0]))
        gripper_2_pos.append(float(row[1]))
        timestamps.append(float(row[2]))

plt.figure(figsize=(15,9))

# Plot all points
plt.plot(timestamps, gripper_1_pos, color='g', linestyle="solid",
         label="Gripper 1 position across time")
plt.plot(timestamps, gripper_2_pos, color='r', linestyle="solid",
         label="Gripper 2 position across time")

# Y-axis: steps of 5, with ±5 buffer
y_min = min(min(gripper_1_pos), min(gripper_2_pos)) - 5
y_max = max(max(gripper_1_pos), max(gripper_2_pos)) + 5
plt.ylim(y_min, y_max)
plt.yticks(np.arange(y_min, y_max + 0.1, 5), fontsize=25)  # 0.1 to include top edge

# X-axis: keep all points, set ticks every 10
x_min = min(timestamps)
x_max = max(timestamps)
plt.xlim(x_min, x_max)
plt.xticks(np.arange(x_min, x_max + 0.1, 10), fontsize=25)  # tick interval only

plt.xlabel('Time (s)', fontsize=35)
plt.ylabel('Gripper opening in mm', fontsize=35)
plt.grid()
plt.legend(fontsize=25)
plt.savefig("corn_50mm_double_mpc_position_plot.pdf", bbox_inches="tight")

