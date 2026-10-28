# 🧮 Machine Learning Foundations: Linear Algebra & Pure NumPy Image Processing

Ushbu repozitoriy **Machine Learning (ML)** va **Deep Learning**ning matematik poydevori bo'lgan **Chiziqli Algebra (Linear Algebra)** hamda **NumPy** kutubxonasini chuqur o'rganish va amalda qo'llashga bag'ishlangan.

> **Asosiy g'oya:** *"Machine Learning'da hamma narsa bu — massiv (tensor)."* 
> Kompyuter uchun rasm shunchaki tasvir emas, balki `(Balandlik, Kenglik, Kanallar)` o'lchamiga ega bo'lgan 3D tensordir. Ushbu loyihada tayyor kutubxonalardan (OpenCV va h.k.) foydalanmasdan, **faqatgina sof NumPy** va chiziqli algebra amallari yordamida tasvirlarni qayta ishlash dvigatelining 1-qismi noldan qurildi.

---

## 🎯 O'rganilgan Matematik va Texnik Konsepsiyalar (Part 1)

Ushbu bosqichda **hech qanday `for` yoki `while` tsikllarisiz** (100% Vectorization orqali) quyidagi chiziqli algebra tushunchalari amalda qo'llanildi:

1. **Tensors & Shapes (Tensorlar va O'lchamlar):**
   - Rangli rasmni `(H, W, 3)` o'lchamli 3D tensor sifatida, oq-qora rasmni esa `(H, W)` o'lchamli 2D matritsa sifatida tahlil qilish.
2. **Multi-dimensional Slicing (Ko'p o'lchamli kesish):**
   - Tensor o'qlari (`axes`) bo'yicha kesish orqali tasvirning kerakli qismini ajratib olish (`crop`) va rang kanallarini (`R, G, B`) 2D matritsalarga bo'lish (`split_channels`).
3. **Axis Manipulations & Negative Strides:**
   - Massiv qatorlari va ustunlarini teskari qadam (`[::-1]`) bilan indekslash orqali gorizontal va vertikal oyna aksini hosil qilish (`flip_horizontal`, `flip_vertical`).
4. **The Dot Product (Skalyar ko'paytma — `@`):**
   - 3D rangli tensorni inson ko'zi sezgirligi og'irliklari vektoriga `[0.299, 0.587, 0.114]` skalyar ko'paytirish orqali haqiqiy oq-qora (grayscale) matritsaga proyeksiya qilish (`to_grayscale`).
5. **Vectorized Arithmetic & Broadcasting (Vektor arifmetikasi):**
   - Skalyar qo'shish va ko'paytirish orqali rasm yorqinligi (`adjust_brightness`), kontrasti (`adjust_contrast`) va negativini (`invert`) hisoblash.
   - `uint8` (0–255) ma'lumot turidagi **Integer Overflow** (sonlar to'lib ketishi) muammosini `float64` ga o'tkazish va `np.clip()` funksiyasi yordamida hal qilish.

---

## 🛠️ Yozilgan Funksiyalar (`imageops.py`)

| Funksiya nomi | Vazifasi | Qo'llanilgan Chiziqli Algebra / NumPy usuli |
| :--- | :--- | :--- |
| `crop(img, top, left, h, w)` | Tasvirning berilgan to'rtburchak qismini kesib olish | 2D/3D Array Slicing |
| `flip_horizontal(img)` | Tasvirni chapdan o'ngga (oyna aksi) ag'darish | Ustunlar o'qi bo'yicha `[:, ::-1]` |
| `flip_vertical(img)` | Tasvirni tepadan pastga ag'darish | Qatorlar o'qi bo'yicha `[::-1, :]` |
| `split_channels(img)` | RGB rasmni 3 ta alohida 2D matritsaga ajratish | Kanal o'qi indeksatsiyasi `[:, :, c]` |
| `to_grayscale(img)` | Rangli (3D) rasmni oq-qora (2D) matritsaga o'tkazish | Skalyar ko'paytma (`Dot Product @`) |
| `adjust_brightness(img, delta)`| Yorqinlikni oshirish yoki kamaytirish | Vektor qo'shish + `np.clip(0, 255)` |
| `adjust_contrast(img, factor)` | 128 markazga nisbatan kontrastni o'zgartirish | Chiziqli transformatsiya `128 + (X - 128) * f` |
| `invert(img)` | Tasvir ranglarini teskarisiga (negativga) o'girish | `255 - X` matritsa ayirmasi |

---

## 🚀 Keyingi Qadam (In Progress — Part 2)
* **2D Convolution Engine (`convolve.py`):** 2D konvolyutsiya (cross-correlation) algoritmini noldan yozish va *Box Blur, Gaussian Blur, Sharpen* hamda *Sobel Edge Detection* (chekkalarni aniqlash) yadrolarini (kernels) qo'llash.

---

## 💻 Ishlatilgan Texnologiyalar
* **Python 3.x**
* **NumPy** (Barcha matematik va matritsa hisob-kitoblari uchun)
* **Pillow (PIL)** (Faqatgina rasmni yuklash uchun)
* **Matplotlib** (Natijalarni vizualizatsiya qilish uchun)
