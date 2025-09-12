import matplotlib.pyplot as plt
import csv

epoch_num = []
val_losses = []
train_losses = []

min_val_loss = [100, -1]
min_train_loss = [100, -1]

with open('./loss_log.csv', 'r') as f:
    reader = csv.reader(f)

    for row in reader:
        epoch_num.append(int(row[0]))
        val_losses.append(float(row[1]))
        train_losses.append(float(row[2]))

        if (float(row[1]) < min_val_loss[0]):
            min_val_loss[0] = float(row[1])
            min_val_loss[1] = int(row[0])

        if (float(row[2]) < min_train_loss[0]):
            min_train_loss[0] = float(row[2])
            min_train_loss[1] = int(row[0])

print(f"Minimum training loss found was {min_train_loss[0]} at epoch {min_train_loss[1]}")
print(f"Minimum validation loss found was {min_val_loss[0]} at epoch {min_val_loss[1]}")

plt.figure(figsize=(15,9))
plt.plot(epoch_num, val_losses, color = 'g', linestyle="solid",
         label = "Validation losses")
plt.plot(epoch_num, train_losses, color = 'r', linestyle="solid",
         label = "Training losses")
plt.xticks(rotation = 25, fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Epoch #', fontsize=35)
plt.ylabel('Loss', fontsize=35)
plt.grid()
plt.legend(fontsize = 25)
plt.savefig("losses_plot.pdf", bbox_inches="tight")