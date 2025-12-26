import os

base_dir = "data/train"

normal_dir = os.path.join(base_dir, "NORMAL")
pneumonia_dir = os.path.join(base_dir, "PNEUMONIA")

normal_files = [f for f in os.listdir(normal_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
pneumonia_files = [f for f in os.listdir(pneumonia_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

print("NORMAL 개수:", len(normal_files))
print("PNEUMONIA 개수:", len(pneumonia_files))
print("총 이미지 개수:", len(normal_files) + len(pneumonia_files))


