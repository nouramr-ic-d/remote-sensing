import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
import rasterio
from rasterio.transform import from_origin
import joblib
import csv
import sys

# =====================================
# 1) LOAD IMAGE
# =====================================

print("Loading STACK.dat ...")

raw = np.fromfile('STACK.dat', dtype=np.float32)

ROWS = 2000
COLS = 2000

BANDS = raw.size // (ROWS * COLS)

print("Bands detected:", BANDS)

full_image = raw.reshape((BANDS, ROWS, COLS))

# =====================================
# 2) LOAD ROI FILE
# =====================================

csv_filename = 'manmona10.csv'

print(f"Loading ROI file: {csv_filename} ...")

pixel_rows = []

with open(csv_filename, 'r', encoding='utf-8', errors='ignore') as f:

    reader = csv.reader(f)

    for r in reader:

        if not r:
            continue

        line_text = " ".join(r).strip()

        if any(k in line_text.lower() for k in
               ["roi name", "roi rgb", "roi npts",
                "file x", "number of", "npts", "color"]):
            continue

        tokens = line_text.replace(',', ' ').split()

        nums = []

        for t in tokens:
            try:
                nums.append(float(t))
            except ValueError:
                pass

        if len(nums) >= 2:
            pixel_rows.append(nums)

print(f"Total pixel data rows found in CSV: {len(pixel_rows)}")

# =====================================
# 3) DISTRIBUTE CLASSES
# =====================================

npts_water = 651
npts_veg = 1943
npts_desert = 1010
npts_urban = 721

X = []
y = []

for idx, nums in enumerate(pixel_rows):

    if idx < npts_water:
        current_class = 0  # Water

    elif idx < (npts_water + npts_veg):
        current_class = 1  # Vegetation

    elif idx < (npts_water + npts_veg + npts_desert):
        current_class = 2  # Desert

    else:
        current_class = 3  # Urban

    c = int(nums[0])
    r_idx = int(nums[1])

    if 0 <= r_idx < ROWS and 0 <= c < COLS:

        X.append(full_image[:, r_idx, c])
        y.append(current_class)

X = np.array(X)
y = np.array(y)

print("\n===============================")
print("Data Extraction Results:")
print("Total Samples collected:", len(X))

if len(y) > 0:

    classes, counts = np.unique(y, return_counts=True)

    class_names = {
        0: "Water",
        1: "Vegetation",
        2: "Desert",
        3: "Urban"
    }

    for cl, co in zip(classes, counts):
        print(f"{class_names[cl]} ({cl}): {co} pixels")

print("===============================\n")

if len(np.unique(y)) < 2:
    print("Error: Failed to separate classes.")
    sys.exit()

# =====================================
# 4) PREPROCESSING
# =====================================

imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

X = imputer.fit_transform(X)
X = scaler.fit_transform(X)

# =====================================
# 5) TRAIN MODEL
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# =====================================
# SAVE MODEL & SCALER & IMPUTER
# =====================================

joblib.dump(model,   'best_land_cover_model.joblib')
joblib.dump(scaler,  'data_scaler.pkl')
joblib.dump(imputer, 'data_imputer.pkl')          # ← السطر الجديد

print("✔️ Model, scaler, and imputer saved.")

# =====================================
# 6) EVALUATION
# =====================================

acc = model.score(X_test, y_test)

print(f"FINAL ACCURACY: {acc}\n")

y_pred = model.predict(X_test)

print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1, 2, 3],
    target_names=["Water", "Vegetation", "Desert", "Urban"],
    zero_division=0
))

# =====================================
# 7) FULL IMAGE CLASSIFICATION
# =====================================

print("Classifying full image...")

flat = full_image.reshape(BANDS, -1).T

flat = imputer.transform(flat)
flat = scaler.transform(flat)

pred = model.predict(flat)

result = pred.reshape(ROWS, COLS).astype(float)

# =====================================
# REMOVE LEFT BLUE TRIANGLE ONLY
# =====================================

for r in range(1400):

    limit = int(220 - (r * 0.11))

    if limit > 0:

        mask = result[r, :limit] == 0

        result[r, :limit][mask] = np.nan

# =====================================
# 8) SAVE PNG RESULT
# =====================================

cmap = ListedColormap([
    "blue",     # Water
    "green",    # Vegetation
    "yellow",   # Desert
    "red"       # Urban
])

cmap.set_bad(color='white')

plt.figure(figsize=(12, 12))

plt.imshow(result, cmap=cmap, vmin=0, vmax=3)

plt.axis("off")

plt.title("Land Cover Classification Map")

plt.savefig(
    "result.png",
    bbox_inches="tight",
    pad_inches=0
)

plt.show()

print("✔️ result.png saved successfully!")

# =====================================
# 9) SAVE TIFF FOR WEB APP
# =====================================

print("Saving classification GeoTIFF...")

tiff_result = np.nan_to_num(result, nan=0).astype(np.uint8)

with rasterio.open(
    "classification_result.tif",
    "w",
    driver="GTiff",
    height=ROWS,
    width=COLS,
    count=1,
    dtype=tiff_result.dtype,
    crs="EPSG:4326",
    transform=from_origin(0, 0, 1, 1)
) as dst:

    dst.write(tiff_result, 1)

print("✔️ classification_result.tif saved successfully!")

print("\nDone ✔️")
