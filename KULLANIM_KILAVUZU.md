# 📚 Neural Network Visualizer - Detaylı Kullanım Kılavuzu

## 🚀 Hızlı Başlangıç

Uygulamayı çalıştırmak için:
```bash
python main.py
```

## 🎯 Temel Kullanım

### 1. Veri Ekleme
- Sol paneldeki "Eğitim (Train)" sekmesinde grafiğe tıklayarak veri ekleyin
- Sağ panelden hangi sınıfa nokta ekleyeceğinizi seçin
- En az 10 veri noktası eklemeniz önerilir

### 2. Model Seçimi
- **Perceptron**: Doğrusal ayrılabilir problemler için
- **Delta Rule**: MSE minimize eden algoritma
- **MLP**: Non-linear problemler için backpropagation

### 3. Hiperparametre Ayarları

#### MLP için:
- **Katman Mimarisi**: `2,8,3` (girdi, gizli, çıktı)
- **Aktivasyon**: `relu,softmax`
- **Öğrenme Oranı**: 0.01 - 0.1 arası
- **Epochs**: 100-500 arası
- **Batch Size**: 16-32 arası

### 4. Eğitim
- **START TRAINING** butonuna tıklayın
- Eğitim sırasında karar sınırlarını izleyin
- "Hata Grafiği" sekmesinde loss değişimini takip edin
- Eğitim bitince "Test" sekmesinde sonuçları görün

## 💡 İpuçları

### Veri Hazırlama
- Sınıflar arasında dengeli sayıda nokta ekleyin
- Farklı bölgelere dağıtarak gerçekçi veri oluşturun

### Model Seçimi
- Basit problemler → Perceptron veya Delta Rule
- XOR, daireler gibi non-linear → MLP
- Overfitting varsa → L2 Regularization artırın

### Eğitim Sorunları
- **Loss azalmıyor**: Öğrenme oranını artırın veya epochs artırın
- **Loss sallanıyor**: Öğrenme oranını düşürün
- **Overfitting**: L2 regularization ekleyin, test split artırın

## 🔧 Örnek Senaryolar

### Senaryo 1: XOR Problemi
```
1. İki sınıf oluşturun
2. Veri: (2,2)→Class0, (8,8)→Class0, (2,8)→Class1, (8,2)→Class1
3. Model: Multi-Layer (MLP)
4. Mimari: 2,8,2
5. Aktivasyon: relu,softmax
6. Öğrenme Oranı: 0.1
7. Epochs: 300
```

### Senaryo 2: 3 Sınıflı Classification
```
1. Üç sınıf oluşturun
2. Her sınıftan 15-20 nokta ekleyin
3. Model: Multi-Layer (MLP)
4. Mimari: 2,10,3
5. Aktivasyon: tanh,softmax
6. Öğrenme Oranı: 0.05
7. Epochs: 200
```

## 📊 Görselleştirme Sekmeleri

- **Eğitim**: Veri ekleme ve eğitim animasyonu
- **Test**: Model performansı ve test accuracy
- **Hata Grafiği**: Loss değişimi (epoch bazlı)

## ⚡ Kısayollar

- Hızlı test için varsayılan ayarları kullanın
- Farklı aktivasyon fonksiyonlarını deneyin
- Learning rate'i loss grafiğine göre ayarlayın

---

**Başarılı eğitimler dileriz! 🚀**
