# 🧠 Neural Network Visualizer - MLP From Scratch

**Profesyonel, interaktif sinir ağı görselleştirme uygulaması**

Python ve NumPy kullanılarak sıfırdan yazılmış tek katmanlı ve çok katmanlı sinir ağı algoritmalarını görselleştiren modern masaüstü uygulaması.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-green.svg)
![CustomTkinter](https://img.shields.io/badge/CustomTkinter-5.2+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 İçindekiler

- [Özellikler](#-özellikler)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Proje Yapısı](#-proje-yapısı)
- [Algoritmalar](#-algoritmalar)
- [Ekran Görüntüleri](#-ekran-görüntüleri)
- [Teknoloji Stack](#-teknoloji-stack)
- [Katkıda Bulunma](#-katkıda-bulunma)

## ✨ Özellikler

### 🎯 İnteraktif Veri Ekleme
- Fare ile doğrudan grafik üzerine tıklayarak veri noktaları ekleyin
- Çoklu sınıf desteği (maksimum 6 sınıf)
- Dinamik sınıf yönetimi

### 🤖 Üç Farklı Algoritma
1. **Single-Layer Perceptron**: Doğrusal olarak ayrılabilir problemler için
2. **Single-Layer Delta Rule (Adaline)**: MSE minimize eden Widrow-Hoff öğrenme kuralı
3. **Multi-Layer Perceptron (MLP)**: Backpropagation ile eğitilen derin sinir ağı

### 🎨 Canlı Görselleştirme
- **Eğitim Sekmesi**: Eğitim sırasında karar sınırlarının canlı animasyonu
- **Test Sekmesi**: Test verisi üzerinde model performansı
- **Hata Grafiği**: Epoch'lara göre loss değişimi

### ⚙️ Esnek Hiperparametre Kontrolü
- Özelleştirilebilir katman mimarisi
- Aktivasyon fonksiyonu seçimi (ReLU, Tanh, Sigmoid, Softmax)
- Öğrenme oranı, epoch sayısı, batch size ayarları
- L2 Regularization desteği
- Test/Train split oranı

### 🔬 Sıfırdan Yazılmış Algoritmalar
- **Backpropagation**: Chain rule ile manuel gradyan hesaplama
- **Aktivasyon Fonksiyonları**: ReLU, Tanh, Sigmoid ve türevleri
- **Loss Fonksiyonu**: Cross-Entropy Loss
- **Optimization**: Mini-batch Gradient Descent
- **Regularization**: L2 (Weight Decay)

## 🚀 Kurulum

### Gereksinimler
- Python 3.8 veya üzeri
- pip paket yöneticisi

### Adım 1: Depoyu Klonlayın
```bash
git clone https://github.com/ensaryesir/MLP-From-Scratch-GUI.git
cd MLP-From-Scratch-GUI
```

### Adım 2: Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

### Adım 3: Uygulamayı Çalıştırın
```bash
python main.py
```

## 📖 Kullanım

### 1️⃣ Veri Ekleme
1. Sağ panelde istediğiniz sınıfı seçin
2. Sol taraftaki **Eğitim (Train)** grafiğine fare ile tıklayarak veri noktaları ekleyin
3. Gerekirse **+ Class** butonu ile yeni sınıflar ekleyin

### 2️⃣ Model Seçimi ve Ayarlama
1. **Model Seçimi**: Perceptron, Delta Rule veya MLP seçin
2. **Hiperparametreler**: 
   - MLP için katman mimarisini ayarlayın (örn: `2,5,3`)
   - Aktivasyon fonksiyonlarını seçin (örn: `relu,softmax`)
   - Öğrenme oranı, epoch sayısı ve diğer parametreleri ayarlayın

### 3️⃣ Eğitim
1. **START TRAINING** butonuna tıklayın
2. Eğitim sırasında:
   - Karar sınırlarının nasıl oluştuğunu izleyin
   - **Hata Grafiği** sekmesinde loss değişimini takip edin
3. Eğitim tamamlandığında:
   - **Test** sekmesine geçerek model performansını görün
   - Test accuracy değerini kontrol edin

### 4️⃣ Veri Temizleme
- **Clear Data** butonu ile tüm veri noktalarını silebilirsiniz

## 📁 Proje Yapısı

```
MLP-From-Scratch-GUI/
├── main.py                      # Ana uygulama orkestratörü
├── requirements.txt             # Python bağımlılıkları
├── README.md                    # Proje dokümantasyonu
│
├── algorithms/                  # Sinir ağı algoritmaları
│   ├── __init__.py
│   ├── single_layer.py         # Perceptron ve Delta Rule
│   └── mlp.py                  # Multi-Layer Perceptron + Backpropagation
│
├── gui/                         # Kullanıcı arayüzü bileşenleri
│   ├── __init__.py
│   ├── control_panel.py        # Kontrol paneli widget'ları
│   └── visualization_frames.py # Görselleştirme sekmeleri
│
└── utils/                       # Yardımcı modüller
    ├── __init__.py
    └── data_handler.py         # Veri yönetimi
```

## 🧮 Algoritmalar

### Perceptron
```
Güncelleme Kuralı: w = w + η * (y_true - y_pred) * x
```
- Step aktivasyon fonksiyonu
- Binary ve multi-class classification desteği

### Delta Rule (Adaline)
```
Loss: MSE = (1/n) * Σ(y_true - y_pred)²
Gradient: ∂L/∂w = -(2/n) * X^T * (y_true - y_pred)
```
- Linear aktivasyon fonksiyonu
- Gradient descent ile eğitim

### Multi-Layer Perceptron (MLP)

**Forward Propagation:**
```
Z^[l] = A^[l-1] * W^[l] + b^[l]
A^[l] = activation(Z^[l])
```

**Backpropagation:**
```
dZ^[L] = A^[L] - Y  (son katman)
dW^[l] = (1/m) * A^[l-1]^T * dZ^[l]
db^[l] = (1/m) * Σ(dZ^[l])
dZ^[l-1] = dZ^[l] * W^[l]^T ⊙ g'(Z^[l-1])
```

**Aktivasyon Fonksiyonları:**
- **ReLU**: `f(x) = max(0, x)`, `f'(x) = 1 if x > 0 else 0`
- **Tanh**: `f(x) = tanh(x)`, `f'(x) = 1 - tanh²(x)`
- **Sigmoid**: `f(x) = 1/(1+e^-x)`, `f'(x) = f(x)(1-f(x))`
- **Softmax**: `f(x_i) = e^x_i / Σe^x_j` (multi-class için)

**Loss Fonksiyonu:**
```
Cross-Entropy: L = -(1/m) * Σ Σ y_true * log(y_pred)
L2 Regularization: L_reg = (λ/2m) * Σ||W||²
```

## 📸 Ekran Görüntüleri

*Uygulamayı çalıştırarak interaktif deneyimi kendiniz yaşayın!*

### Özellikler:
- ✅ Modern dark mode arayüz
- ✅ Renkli karar sınırları
- ✅ Real-time animasyonlar
- ✅ Profesyonel grafikler

## 🛠️ Teknoloji Stack

- **Python 3.8+**: Ana programlama dili
- **NumPy**: Sayısal hesaplamalar ve matris işlemleri
- **Matplotlib**: Bilimsel görselleştirme
- **CustomTkinter**: Modern GUI framework

### ⚠️ Yasaklı Kütüphaneler
Bu proje **eğitim amaçlı** olduğundan, aşağıdaki kütüphaneler **KESİNLİKLE KULLANILMAMIŞTIR**:
- ❌ scikit-learn
- ❌ TensorFlow
- ❌ PyTorch
- ❌ Keras

Tüm algoritmalar sıfırdan NumPy ile yazılmıştır.

## 🤝 Katkıda Bulunma

Katkılarınızı bekliyoruz! Lütfen şu adımları izleyin:

1. Projeyi fork edin
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some AmazingFeature'`)
4. Branch'inizi push edin (`git push origin feature/AmazingFeature`)
5. Pull Request açın

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## 👨‍💻 Geliştirici

**Ensar Yeşir**
- GitHub: [@ensaryesir](https://github.com/ensaryesir)

## 🙏 Teşekkürler

Bu proje, makine öğrenmesi ve sinir ağları derslerinde öğrenilen teorik bilgilerin pratik uygulamasıdır.

---

⭐ Projeyi beğendiyseniz yıldız vermeyi unutmayın!

**Keyifli kodlamalar! 🚀**