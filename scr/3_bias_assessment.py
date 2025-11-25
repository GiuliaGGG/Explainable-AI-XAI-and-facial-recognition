from imports import *

classes = sorted([
    d for d in os.listdir(DATA_DIR) 
    if os.path.isdir(os.path.join(DATA_DIR, d))
])

print("Classes detected:")
for c in classes:
    print(" -", c)

num_classes = len(classes)
print("Number of classes:", num_classes)

# 1. Recreate the architecture
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# 2. Load weights (correct path)
model_path = os.path.join("face_classifier_resnet18.pth")
state = torch.load(model_path, map_location="cpu")

# 3. Assign weights
model.load_state_dict(state)

# 4. Put in evaluation mode
model.eval()
print("Model loaded successfully.")

# model structure summary 
summary(model, input_size=(1, 3, 224, 224))

