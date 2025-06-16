import matplotlib.pyplot as plt
import csv

epoch_num = []
val_losses = []
train_losses = []

with open('./50_soft_50_hard_v1_epochs/loss_log.csv', 'r') as f:
    reader = csv.reader(f)

    for row in reader:
        epoch_num.append(int(row[0]))
        val_losses.append(float(row[1]))
        train_losses.append(float(row[2]))

plt.figure(figsize=(15,9))
plt.plot(epoch_num, val_losses, color = 'g', linestyle="solid",
         label = "Validation losses")
plt.plot(epoch_num, train_losses, color = 'r', linestyle="solid",
         label = "Training losses")
plt.xticks(rotation = 25)
plt.xlabel('Epoch #', fontsize=30)
plt.ylabel('Losses', fontsize=30)
plt.title('Losses across epochs', fontsize = 30)
plt.grid()
plt.legend()
plt.savefig("losses_plot.png")