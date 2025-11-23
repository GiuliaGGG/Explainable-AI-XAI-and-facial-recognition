from imports import *

# ----------------------------------------------------
# 1. Load dataset structure
# ----------------------------------------------------
classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
print("Classes detected:")
for c in classes:
    print(" -", c)


# Gather image paths
image_paths = []
labels = []
for cls in classes:
    cls_dir = os.path.join(DATA_DIR, cls)
    for fname in os.listdir(cls_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            image_paths.append(os.path.join(cls_dir, fname))
            labels.append(cls)


print(f"\nTotal images found: {len(image_paths)}")


# ----------------------------------------------------
# 2. Class distribution
# ----------------------------------------------------
class_counts = Counter(labels)
print("\nClass distribution:")
for cls, count in class_counts.items():
    print(f"{cls}: {count}")


# ----------------------------------------------------
# 3. Image dimension statistics
# ----------------------------------------------------
widths = []
heights = []


for path in image_paths[:500]: # sample first 500 images for speed
    try:
        with Image.open(path) as img:
            w, h = img.size
            widths.append(w)
            heights.append(h)
    except Exception:
        pass


print(f"\nSampled {len(widths)} images for dimension analysis.")
print(f"Average width: {sum(widths)/len(widths):.1f}")
print(f"Average height: {sum(heights)/len(heights):.1f}")


# ----------------------------------------------------
# 4. Plot a few example images
# ----------------------------------------------------
plt.figure(figsize=(10, 6))
for i, path in enumerate(image_paths[:6]):
    plt.subplot(2, 3, i + 1)
    img = Image.open(path)
    plt.imshow(img)
    plt.title(labels[i])
    plt.axis("off")
plt.tight_layout()
plt.show()