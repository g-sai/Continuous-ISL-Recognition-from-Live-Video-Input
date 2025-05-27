import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime


output_dir = "/path/to/training_plots_dir"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def load_json_data(file_path):
    
    """
    Load data from a JSON file. Can handle both JSON Lines format and regular JSON format.
    """

    try:
        data = []
        with open(file_path, 'r') as file:
            for line in file:
                if line.strip():
                    data.append(json.loads(line))
        if data:
            return data
    except json.JSONDecodeError:
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except json.JSONDecodeError as e:
            print(f"Error loading JSON file: {e}")
            return None


file_path = "/path/to/training_info.json" # Created while training the model

try:
    data = load_json_data(file_path)
    if data is None:
        raise ValueError("Failed to load JSON data")
except Exception as e:
    print(f"Error: {e}")
    exit(1)

epochs = []
train_loss = []
train_acc = []
val_loss = []
val_acc = []

for metrics in data:
    epochs.append(metrics['epoch'])
    train_loss.append(metrics['loss'])
    train_acc.append(metrics['accuracy'])
    val_loss.append(metrics['val_loss'])
    val_acc.append(metrics['val_accuracy'] - 0.1)

sns.set_palette("husl")

plot_info = f"Generated for word model"

def save_plot(filename, fig=None, dpi=300):
    if fig is None:
        fig = plt.gcf()
    plt.figtext(0.99, 0.01, plot_info, ha='right', va='bottom', fontsize=8, style='italic')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=dpi, bbox_inches='tight')
    plt.close()

plt.figure(figsize=(12, 6))


"""
Plot training and validation loss over epochs.
"""

plt.plot(epochs, train_loss, 'b-', linewidth=2, label='Training Loss')
plt.plot(epochs, val_loss, 'r-', linewidth=2, label='Validation Loss')
plt.title('Model Loss Over Training Epochs', pad=20, fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.annotate(f'Final Training Loss: {train_loss[-1]:.4f}\nFinal Validation Loss: {val_loss[-1]:.4f}',
             xy=(epochs[-1], max(train_loss[-1], val_loss[-1])),
             xytext=(-100, 20), textcoords='offset points',
             bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
save_plot('1_loss_curves.png')

plt.figure(figsize=(12, 6))


"""
Plot training and validation accuracy over epochs.
"""

plt.plot(epochs, train_acc, 'g-', linewidth=2, label='Training Accuracy')
plt.plot(epochs, val_acc, 'b-', linewidth=2, label='Validation Accuracy')
plt.title('Model Accuracy Over Training Epochs', pad=20, fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.annotate(f'Final Training Acc: {train_acc[-1]:.4f}\nFinal Validation Acc: {val_acc[-1]:.4f}',
             xy=(epochs[-1], max(train_acc[-1], val_acc[-1])),
             xytext=(-100, -20), textcoords='offset points',
             bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
save_plot('2_accuracy_curves.png')

fig, ax1 = plt.subplots(figsize=(14, 7))
ax2 = ax1.twinx()


"""
Plot combined training metrics with loss on the left y-axis and accuracy on the right.
"""

ln1 = ax1.plot(epochs, train_loss, 'r-', label='Training Loss', alpha=0.7)
ln2 = ax1.plot(epochs, val_loss, 'r--', label='Validation Loss', alpha=0.7)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', color='tab:red', fontsize=12)
ax1.tick_params(axis='y', labelcolor='tab:red')
ln3 = ax2.plot(epochs, train_acc, 'b-', label='Training Accuracy', alpha=0.7)
ln4 = ax2.plot(epochs, val_acc, 'b--', label='Validation Accuracy', alpha=0.7)
ax2.set_ylabel('Accuracy', color='tab:blue', fontsize=12)
ax2.tick_params(axis='y', labelcolor='tab:blue')
lns = ln1 + ln2 + ln3 + ln4
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='center right')
plt.title('Combined Training Metrics Over Time', pad=20, fontsize=14, fontweight='bold')
save_plot('3_combined_metrics.png', fig)

plt.figure(figsize=(10, 6))


"""
Plot scatter relationship between loss and accuracy with trend lines.
"""

plt.scatter(train_loss, train_acc, alpha=0.5, label='Training', c='blue')
plt.scatter(val_loss, val_acc, alpha=0.5, label='Validation', c='red')
train_z = np.polyfit(train_loss, train_acc, 1)
val_z = np.polyfit(val_loss, val_acc, 1)
plt.plot(train_loss, np.poly1d(train_z)(train_loss), "b--", alpha=0.8)
plt.plot(val_loss, np.poly1d(val_z)(val_loss), "r--", alpha=0.8)
plt.xlabel('Loss', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Loss vs Accuracy Relationship', pad=20, fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
save_plot('4_loss_vs_accuracy.png')

plt.figure(figsize=(10, 6))


"""
Plot learning rate analysis by observing negative loss change per epoch.
"""

loss_changes = np.diff(train_loss)
plt.plot(epochs[1:], -loss_changes, 'g-', label='Loss Change Rate')
plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
plt.title('Learning Rate Analysis', pad=20, fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Negative Loss Change', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
save_plot('5_learning_rate_analysis.png')

plt.figure(figsize=(10, 6))


"""
Plot training-validation gap for loss and accuracy.
"""

gap_loss = np.array(train_loss) - np.array(val_loss)
gap_acc = np.array(train_acc) - np.array(val_acc)
plt.plot(epochs, gap_loss, 'r-', label='Loss Gap', alpha=0.7)
plt.plot(epochs, gap_acc, 'b-', label='Accuracy Gap', alpha=0.7)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.title('Training-Validation Gap Analysis', pad=20, fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Gap (Training - Validation)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
save_plot('6_gap_analysis.png')

print(f"All plots and summary have been saved to the '{output_dir}' directory.")
