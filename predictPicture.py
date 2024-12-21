from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO(r"C:\Tugas Sekolah dan Kuliah\Universitas Udayana\SEMESTER 3\Pengolahan Citra Digital\Tugas Akhir\persiapan_uas\train\weights\best.pt")

# Run inference on 'bus.jpg' with arguments
model.predict(r"C:\Tugas Sekolah dan Kuliah\Universitas Udayana\SEMESTER 3\Pengolahan Citra Digital\Tugas Akhir\persiapan_uas\gambarBotol.jpg", save=True, conf=0.5)