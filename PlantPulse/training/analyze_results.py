"""
Analyze training results for overfitting and performance issues
"""

import matplotlib.pyplot as plt
import numpy as np

# Training metrics from the output
epochs = list(range(1, 11))

# Training accuracy progression (final values per epoch)
train_disease_acc = [0.2390, 0.2700, 0.2570, 0.2490, 0.2850, 0.3170, 0.4010, 0.4710, 0.5660, 0.5560]
val_disease_acc = [0.2700, 0.2550, 0.2700, 0.2100, 0.2550, 0.2650, 0.2650, 0.2700, 0.2650, 0.2700]

# Training loss progression
train_loss = [2.1073, 1.7312, 1.6675, 1.6223, 1.5994, 1.5541, 1.4748, 1.3362, 1.0999, 1.0859]
val_loss = [1.8695, 1.7114, 1.6313, 1.6282, 1.6387, 2.1966, 2.3247, 12.5113, 17.7409, 8.0065]

# Disease loss specifically
train_disease_loss = [1.5431, 1.4483, 1.4270, 1.3987, 1.3897, 1.3494, 1.2747, 1.1337, 0.9012, 0.8942]
val_disease_loss = [1.4074, 1.4755, 1.4212, 1.4087, 1.4106, 1.9451, 2.0939, 12.0990, 17.4527, 7.7258]

print("🔍 OVERFITTING ANALYSIS")
print("=" * 50)

print("\n1. ACCURACY DIVERGENCE:")
print(f"   Training accuracy: {train_disease_acc[0]:.1%} → {train_disease_acc[-1]:.1%} (+{(train_disease_acc[-1] - train_disease_acc[0]):.1%})")
print(f"   Validation accuracy: {val_disease_acc[0]:.1%} → {val_disease_acc[-1]:.1%} ({(val_disease_acc[-1] - val_disease_acc[0]):+.1%})")
print(f"   \n   ⚠️  Training improved by {(train_disease_acc[-1] - train_disease_acc[0]):.1%}")
print(f"   ⚠️  Validation stayed flat at ~27%")
print(f"   ⚠️  Gap: {(train_disease_acc[-1] - val_disease_acc[-1]):.1%}")

print("\n2. LOSS EXPLOSION:")
print("   Validation loss progression:")
for i, (epoch, loss) in enumerate(zip(epochs, val_loss)):
    marker = "💥" if loss > 5 else "📈" if loss > 2 else "→"
    print(f"   Epoch {epoch:2d}: {loss:6.2f} {marker}")

print("\n3. CLEAR SIGNS OF OVERFITTING:")
print("   ✗ Training accuracy improving while validation stuck")
print("   ✗ Validation loss exploding (up to 17.74!)")
print("   ✗ Model memorizing training data, not generalizing")
print("   ✗ 27% validation accuracy ≈ random guessing (25% for 4 classes)")

print("\n4. ROOT CAUSES:")
print("   • Synthetic data too simple/repetitive")
print("   • Model architecture too complex for data")
print("   • No regularization techniques applied")
print("   • Small dataset (1000 train, 200 val)")

print("\n5. RECOMMENDATIONS:")
print("   • Add dropout layers (already have 0.5, maybe increase)")
print("   • Use data augmentation")
print("   • Reduce model complexity")
print("   • Add L2 regularization")
print("   • Use early stopping (val loss increased after epoch 3)")
print("   • Get real thermal data for better patterns")

# Create visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Accuracy plot
ax1.plot(epochs, train_disease_acc, 'b-', label='Train', linewidth=2)
ax1.plot(epochs, val_disease_acc, 'r-', label='Validation', linewidth=2)
ax1.axhline(y=0.25, color='gray', linestyle='--', label='Random (25%)')
ax1.set_title('Disease Classification Accuracy', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Loss plot
ax2.plot(epochs, train_loss, 'b-', label='Train', linewidth=2)
ax2.plot(epochs, val_loss, 'r-', label='Validation', linewidth=2)
ax2.set_title('Total Loss (Note Explosion!)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_yscale('log')  # Log scale due to explosion
ax2.legend()
ax2.grid(True, alpha=0.3)

# Disease loss specifically
ax3.plot(epochs, train_disease_loss, 'b-', label='Train', linewidth=2)
ax3.plot(epochs, val_disease_loss, 'r-', label='Validation', linewidth=2)
ax3.set_title('Disease Classification Loss', fontsize=14, fontweight='bold')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss')
ax3.set_yscale('log')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Overfitting indicator
overfitting_gap = [t - v for t, v in zip(train_disease_acc, val_disease_acc)]
ax4.bar(epochs, overfitting_gap, color=['red' if gap > 0.2 else 'orange' if gap > 0.1 else 'green' for gap in overfitting_gap])
ax4.axhline(y=0.2, color='red', linestyle='--', label='Severe overfitting')
ax4.axhline(y=0.1, color='orange', linestyle='--', label='Moderate overfitting')
ax4.set_title('Overfitting Gap (Train - Val Accuracy)', fontsize=14, fontweight='bold')
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Accuracy Gap')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.suptitle('🚨 SEVERE OVERFITTING DETECTED 🚨', fontsize=16, fontweight='bold', color='red')
plt.tight_layout()
plt.savefig('overfitting_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n📊 Plot saved as 'overfitting_analysis.png'")

# Calculate some statistics
print("\n📈 FINAL STATISTICS:")
print(f"   Best validation accuracy: {max(val_disease_acc):.1%} (epoch {val_disease_acc.index(max(val_disease_acc)) + 1})")
print(f"   Best validation loss: {min(val_loss):.3f} (epoch {val_loss.index(min(val_loss)) + 1})")
print(f"   Should have stopped at: Epoch 3 (before val loss explosion)")
print(f"   Actual performance: Barely better than random guessing!")

print("\n⚠️  CONCLUSION: This model is NOT suitable for deployment!")
print("   It has memorized the training data and cannot generalize.")