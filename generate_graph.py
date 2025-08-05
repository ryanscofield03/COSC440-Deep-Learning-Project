import os

import matplotlib.pyplot as plt
PATH = "./data/model2_conv_and_attention_normed_with_gaussian_loss/lambda1"


epoch_data_dict = {}
with open(os.path.join(PATH, "output.txt"), "r") as f:
    for line in f:
        if line.startswith("EPOCH:"):
            try:
                # Split into components
                parts = line.strip().split()

                # Extract values - using fixed positions since format is consistent
                epoch = int(parts[1])
                loss = float(parts[3])
                ssim = float(parts[5])
                psnr = float(parts[7])

                epoch_data_dict[epoch] = {
                    'loss': loss,
                    'ssim': ssim,
                    'psnr': psnr
                }
            except (IndexError, ValueError) as e:
                print(f"Error parsing line: {line.strip()}")
                print(f"Error: {e}")
                continue

# Convert to lists for plotting
epochs = sorted(epoch_data_dict.keys())
loss = [epoch_data_dict[e]['loss'] for e in epochs]
ssim = [epoch_data_dict[e]['ssim'] for e in epochs]
psnr = [epoch_data_dict[e]['psnr'] for e in epochs]

plt.figure(figsize=(10, 5))
fig1, ax1 = plt.subplots(figsize=(10, 5))

ax1.set_xlabel('Epoch')
ax1.set_ylabel('SSIM', color='blue')
ax1.plot(epochs, ssim, color='blue', label='SSIM')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.set_ylabel('PSNR (dB)', color='green')
ax2.plot(epochs, psnr, color='green', label='PSNR')
ax2.tick_params(axis='y', labelcolor='green')

plt.title('SSIM and PSNR over Training')
fig1.tight_layout()
plt.savefig(os.path.join(PATH, 'ssim_psnr_plot.png'))
plt.close()

# Plot 2: Loss (log scale)
plt.figure(figsize=(10, 5))
fig2, ax3 = plt.subplots(figsize=(10, 5))

ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss (log scale)')
ax3.set_yscale('log')
ax3.plot(epochs, loss, color='red', label='Loss')
ax3.grid(True, which="both", ls="-")

plt.title('Training Loss in Logarithmic Scale')
fig2.tight_layout()
plt.savefig(os.path.join(PATH, 'loss_plot_log.png'))
plt.close()
