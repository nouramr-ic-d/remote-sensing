import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import joblib

# =====================================
# 1) LOAD MODEL + SCALER + IMPUTER
# =====================================

print("Loading model, scaler, and imputer...")

model   = joblib.load('best_land_cover_model.joblib')
scaler  = joblib.load('data_scaler.pkl')
imputer = joblib.load('data_imputer.pkl')

print("All files loaded successfully.")

# =====================================
# 2) LOAD NEW IMAGE (10 bands - same format as training)
# =====================================

image_path = 'Marcello.dat'   # <-- change to your new image filename

print(f"Loading: {image_path} ...")

raw = np.fromfile(image_path, dtype=np.float32)   # same dtype as training

ROWS  = 2000
COLS  = 2000
BANDS = raw.size // (ROWS * COLS)

print(f"Bands detected: {BANDS}")

full_image = raw.reshape((BANDS, ROWS, COLS))

# Print value range to verify correct loading
print(f"Min value  : {full_image.min():.4f}")
print(f"Max value  : {full_image.max():.4f}")
print(f"Mean value : {full_image.mean():.4f}")

# =====================================
# 3) FLATTEN - same as training
# =====================================

print("Preprocessing...")

flat = full_image.reshape(BANDS, -1).T    # shape: (4000000, 10)

flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)

flat = imputer.transform(flat)
flat = scaler.transform(flat)

print(f"Feature matrix shape: {flat.shape}")

# =====================================
# 4) CLASSIFY
# =====================================

print("Classifying full image...")

pred   = model.predict(flat)
result = pred.reshape(ROWS, COLS).astype(float)

# Print class distribution
unique, counts = np.unique(pred, return_counts=True)
class_names = {0: 'Water', 1: 'Vegetation', 2: 'Desert', 3: 'Urban'}
print("\nClass distribution:")
for u, c in zip(unique, counts):
    pct = (c / len(pred)) * 100
    print(f"  {class_names.get(int(u), u)}: {c:,} pixels ({pct:.1f}%)")

# =====================================
# 5) SAVE PNG
# =====================================

cmap = ListedColormap(["blue", "green", "yellow", "red"])
cmap.set_bad(color='white')

legend = [
    Patch(color='blue',   label='Water'),
    Patch(color='green',  label='Vegetation'),
    Patch(color='yellow', label='Desert'),
    Patch(color='red',    label='Urban'),
]

plt.figure(figsize=(12, 12))
plt.imshow(result, cmap=cmap, vmin=0, vmax=3)
plt.legend(handles=legend, loc='lower right', fontsize=12)
plt.axis("off")
plt.title("Land Cover Classification - New Image")
plt.savefig("result_new.png", bbox_inches="tight", pad_inches=0)
plt.show()

print("result_new.png saved!")
