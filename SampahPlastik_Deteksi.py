import os
import cv2
import shutil
import tempfile
import numpy as np
import tkinter as tk
from ultralytics import YOLO
from tkinter import filedialog
from PIL import Image, ImageTk

# Variabel untuk menyimpan gambar yang dipilih dan ukuran gambar yang diubah
hasil_pengolahan_path = None
temp_file_path = None
selected_path = None
resized_image = None
main_image = None
main_np = None
new_width = None
new_height = None
parameter_proses = -1

def pilih_gambar():
    global selected_path, resized_image, main_image, new_width, new_height, parameter_proses, main_np  # Gunakan variabel global untuk menyimpan gambar dan ukuran

    # Membuka dialog untuk memilih gambar
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.gif;*.bmp")])
    
    if file_path:
        # Membuka gambar
        image = Image.open(file_path)
        
        # Menyimpan gambar yang dipilih
        selected_path = file_path
        
        # Mendapatkan ukuran asli gambar
        original_width, original_height = image.size
        
        # Ukuran maksimum yang diinginkan
        max_width = 800
        max_height = 400
        
        # Menghitung rasio untuk menyesuaikan ukuran
        ratio_width = max_width / original_width
        ratio_height = max_height / original_height
        ratio = min(ratio_width, ratio_height)  # Pilih rasio terkecil untuk menjaga aspek gambar
        
        # Menyesuaikan ukuran gambar
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        
        # Mengubah ukuran gambar dengan algoritma LANCZOS (untuk hasil terbaik)
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        main_image = resized_image
        main_np = np.array(main_image)
        
        # Membuat gambar yang dapat ditampilkan di Tkinter
        photo = ImageTk.PhotoImage(resized_image)
        
        # Mengupdate label untuk menampilkan gambar
        label_preview.config(image=photo)
        label_preview.image = photo
        label_status.config(text="Gambar berhasil dipilih!")

        # Menampilkan tombol untuk memproses gambar
        btn_proses.grid(row=2, column=0, padx=10, pady=20)

        # Mengembalikan Parameter Proses ke 0
        parameter_proses = 0

def proses_gambar():
    global parameter_proses, selected_path, main_image, main_np
    print(parameter_proses)
    if parameter_proses == 0 :
        parameter_proses = parameter_proses + 1

        # Meningkatkan Kontras Gambar
        main_image = tingkatkan_kontras_rgb()

        # Mengupdate label untuk menampilkan gambar grayscale
        label_preview.config(image=main_image)
        label_preview.image = main_image
        label_status.config(text="Gambar berhasil ditingkatkan kontrasnya!")
    elif parameter_proses == 1:
        parameter_proses = parameter_proses + 1

        # Ambil Path Hasil Pengolahan Terakhir
        simpan_gambar_np()

        # Prediksi Gambar
        prediksi_gambar()

        # Ambil file yang telah diprediksi
        main_image = ambil_gambar_prediksi()

        # Mengupdate label untuk menampilkan gambar grayscale
        label_preview.config(image=main_image)
        label_preview.image = main_image
        label_status.config(text="Gambar berhasil diprediksi!")
    else:
        label_status.config(text="Proses Selesai, Silahkan Pilih Gambar Lain!")

def tingkatkan_kontras_rgb():
    global main_np
    # Mengonversi gambar dari BGR (OpenCV default) ke RGB
    img_rgb = cv2.cvtColor(main_np, cv2.COLOR_BGR2RGB)
    
    # Pisahkan saluran warna
    r, g, b = cv2.split(img_rgb)
    
    # Terapkan CLAHE pada setiap saluran
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(5, 5))
    r_clahe = clahe.apply(r)
    g_clahe = clahe.apply(g)
    b_clahe = clahe.apply(b)
    
    # Gabungkan saluran kembali menjadi gambar berwarna
    enhanced_img = cv2.merge([r_clahe, g_clahe, b_clahe])
    main_np = enhanced_img
    
    # Convert to PIL untuk preview
    pil_image = Image.fromarray(enhanced_img)
    tk_image = ImageTk.PhotoImage(pil_image)
    
    return tk_image

def prediksi_gambar():
    # Ambil Path Folder untuk menyimpan hasil prediksi
    folder_path = r"C:\Tugas Sekolah dan Kuliah\Universitas Udayana\SEMESTER 3\Pengolahan Citra Digital\Tugas Akhir\persiapan_uas\runs\detect\predict"

    # Memeriksa apakah path folder ada
    if os.path.exists(folder_path):
        # Menghapus folder beserta isinya jika ada
        shutil.rmtree(folder_path)
        print(f"Folder {folder_path} telah dihapus.")
    else:
        print(f"Folder {folder_path} tidak ditemukan. (Aman untuk disimpan)")
    
    # Mengambil Model YOLO
    model = YOLO(r"C:\Tugas Sekolah dan Kuliah\Universitas Udayana\SEMESTER 3\Pengolahan Citra Digital\Tugas Akhir\persiapan_uas\train\weights\best.pt")

    # Deteksikan gambar
    model.predict(hasil_pengolahan_path, save=True, conf=0.5)

def ambil_gambar_prediksi():
    # Buka File Gambar
    file_path_predicted = r"C:\Tugas Sekolah dan Kuliah\Universitas Udayana\SEMESTER 3\Pengolahan Citra Digital\Tugas Akhir\persiapan_uas\runs\detect\predict"

    # Mengambil seluruh file dalam folder dan subfolder
    jpg_files = [file for file in os.listdir(file_path_predicted) if file.endswith(".jpg")]

    # Mengambil file gambar pertama
    file_path_predicted = os.path.join(file_path_predicted, jpg_files[0])

    # Membuka gambar
    predicted_image = Image.open(file_path_predicted)

    # Mengubah ukuran gambar dengan algoritma LANCZOS (untuk hasil terbaik)
    resized_image_predicted = predicted_image.resize((new_width, new_height), Image.LANCZOS)
        
    # Membuat gambar yang dapat ditampilkan di Tkinter
    photo_predicted = ImageTk.PhotoImage(resized_image_predicted)

    return photo_predicted

def simpan_gambar_np():
    global main_np, hasil_pengolahan_path
    img_bgr = None
    # Tentukan path penyimpanan
    save_path = r"C:\Tugas Sekolah dan Kuliah\Universitas Udayana\SEMESTER 3\Pengolahan Citra Digital\Tugas Akhir\persiapan_uas\hasil_pengolahan\gambar_hasil.jpg"

    # Mengambil Path Hasil Pengolahan
    hasil_pengolahan_path = save_path

    # Memeriksa apakah path file ada
    if os.path.exists(save_path):
        # Menghapus file jika ada
        os.remove(save_path)
        print(f"File {save_path} telah dihapus.")
    else:
        print(f"File {save_path} tidak ditemukan. (Aman untuk disimpan)")

    # Mengonversi BGR ke RGB jika perlu dan menyimpan gambar
    if len(main_np.shape) == 3 and main_np.shape[2] == 3:  # Pastikan gambar dalam format RGB/BGR
        img_bgr = cv2.cvtColor(main_np, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = main_np  # Jika sudah dalam format BGR, gunakan langsung

    # Simpan gambar
    cv2.imwrite(save_path, img_bgr)
    print(f"Gambar berhasil disimpan di {save_path}")

root = tk.Tk()
root.title("Pendeteksi Sampah Berdasarkan Citra")

# Mendapatkan ukuran layar
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Ukuran jendela yang diinginkan
window_width = 900
window_height = 650

# Menghitung posisi jendela agar berada di tengah layar
x_position = (screen_width - window_width) // 2
y_position = (screen_height - window_height) // 2

# Mengatur posisi jendela dan ukurannya
root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

# Frame untuk layout
frame = tk.Frame(root)
frame.pack(pady=20)

# Tombol untuk memilih gambar
btn_pilih = tk.Button(frame, text="Pilih Gambar", font=("Arial", 14), width=20, bg="#4CAF50", fg="white", command=pilih_gambar)
btn_pilih.grid(row=0, column=0, padx=10, pady=10)

# Label untuk preview gambar
label_preview = tk.Label(frame)
label_preview.grid(row=1, column=0, padx=10, pady=20)

# Tombol untuk memproses gambar (dihide dulu, akan ditampilkan setelah gambar dipilih)
btn_proses = tk.Button(frame, text="Proses Gambar", font=("Arial", 14), width=20, bg="#FF9800", fg="white", command=proses_gambar)

# Label status untuk menampilkan informasi
label_status = tk.Label(root, text="Pilih gambar untuk melihat preview", font=("Arial", 12), fg="gray")
label_status.pack()

root.mainloop()
