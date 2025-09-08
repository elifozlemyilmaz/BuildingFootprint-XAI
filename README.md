# 🏙️ BuildingFootprint-XAI  
*Açıklanabilir Yapay Zekâ ile Çok Yüksek Çözünürlüklü Görüntülerden Bina Çıkarımı*  

---

## 📑 İçindekiler / Table of Contents  
- [🚀 Kısa Özet / Summary](#-kısa-özet--summary)  
- [🛰️ Kullanılan Veri / Data Used](#-kullanılan-veri--data-used)  
- [🧠 Modeller / Models](#-modeller--models)  
- [🔎 XAI Yöntemleri / XAI-Methods](#-xai-yöntemleri--xai-methods)  
- [⚙️ Nasıl Kullanılır? / How to Use](#-nasıl-kullanılır--how-to-use)  
- [📊 Sonuçlar / Results](#-sonuçlar--results)  
- [📜 Lisans / License](#-lisans--license)  

---

## 🚀 Kısa Özet / Summary  

🇹🇷 **Türkçe:**  
Bu proje, TÜBİTAK 1002 desteğiyle yürütülmüş olup, çok yüksek çözünürlüklü uydu ve hava görüntülerinden bina ayak izlerinin otomatik çıkarımı üzerine odaklanmaktadır. Çalışmada **U-Net++**, **DeepLabV3+**, **PSPNet** ve **DeepSwinLite** gibi güncel derin öğrenme modelleri uygulanmış; ayrıca açıklanabilir yapay zekâ (XAI) yöntemleri kullanılarak bu modellerin karar mekanizmaları incelenmiştir. Elde edilen sonuçlar, farklı kentsel dokularda performans farklılıklarını ortaya koymakta ve yöntemlerin **şehir planlama, afet yönetimi ve sürdürülebilir şehircilik** uygulamalarında güvenilir bir şekilde kullanılabileceğini göstermektedir.  

🇬🇧 **English:**  
This project, supported by TÜBİTAK 1002, focuses on the automatic extraction of building footprints from very high-resolution satellite and aerial imagery. State-of-the-art deep learning models such as **U-Net++**, **DeepLabV3+**, **PSPNet**, and **DeepSwinLite** were applied, and explainable artificial intelligence (XAI) techniques were employed to analyze the decision-making processes of these models. The results reveal performance differences across various urban patterns and demonstrate that the proposed approaches can be reliably used in applications such as **urban planning, disaster management, and sustainable city development**.  

---

## 🛰️ Kullanılan Veri / Data Used  

🇹🇷 **Türkçe:**  
Projede iki farklı veri seti kullanılmıştır: **SPOT 6/7 (Fransa, Pyrénées-Orientales)** ve **MAXAR (Türkiye, İzmir)**. Bu veriler çok yüksek çözünürlüklü olup farklı kentsel dokuları temsil etmektedir. Görüntülerin yanı sıra, bina sınırlarını içeren **shapefile (vektör) verileri** de kullanılmıştır:  

- Fransa / Pyrénées-Orientales bina sınırları shapefile → *IGN (Institut National de l'Information Géographique et Forestière)* → [https://geoservices.ign.fr/](https://geoservices.ign.fr/)  
- Türkiye / İzmir bina sınırları shapefile → *HERE Maps* veritabanı → [https://developer.here.com/](https://developer.here.com/)  

> 📌 **Not:** Lisans kısıtları nedeniyle ham görüntüler ve shapefile verileri bu repoda paylaşılamamaktadır. Araştırmacılar ilgili sağlayıcılardan resmi olarak temin edebilirler.  

🇬🇧 **English:**  
Two datasets were utilized in this project: **SPOT 6/7 (France, Pyrénées-Orientales)** and **MAXAR (Turkey, Izmir)**. These very high-resolution datasets represent diverse urban patterns. In addition to imagery, **shapefile (vector) building footprint data** were also used:  

- France / Pyrénées-Orientales building footprints → *IGN (Institut National de l'Information Géographique et Forestière)* → [https://geoservices.ign.fr/](https://geoservices.ign.fr/)  
- Turkey / Izmir building footprints → *HERE Maps* database → [https://developer.here.com/](https://developer.here.com/)  

> 📌 **Note:** Due to licensing restrictions, the raw imagery and shapefile data cannot be shared in this repository. Researchers can obtain them officially from the respective providers.  

---

## 🧠 Modeller / Models  

| Model          | Açıklama (TR) | Description (EN) |
|----------------|---------------|------------------|
| **U-Net++**    | Gelişmiş U-Net, skip-connection yapısıyla daha hassas segmentasyon sağlar. | Enhanced U-Net with nested skip connections for more accurate segmentation. |
| **DeepLabV3+** | Atrous konvolüsyon + ASPP ile çok ölçekli öğrenme sağlar. | Employs atrous convolution and Atrous Spatial Pyramid Pooling (ASPP) for multi-scale learning. |
| **PSPNet**     | Pyramid Pooling ile bağlamsal bilgi yakalar. | Utilizes Pyramid Pooling Module to capture contextual information. |
| **DeepSwinLite** | Projede geliştirilen hafif Swin Transformer tabanlı model. | Lightweight Swin Transformer-based model developed within this project. |  

---

## 🔎 XAI Yöntemleri / XAI Methods  

🇹🇷 **Türkçe:**  
Projede kullanılan modellerin karar mekanizmalarını analiz edebilmek için aşağıdaki XAI yöntemleri uygulanmıştır:  
- **Saliency Maps** → Modelin en çok dikkate aldığı pikselleri gösterir.  
- **GradientSHAP** → Gradyan tabanlı duyarlılık analizi ile girişin çıktıya etkisini ölçer.  
- **Integrated Gradients (IG)** → Model kararını adım adım izleyerek özelliklerin önemini belirler.  

🇬🇧 **English:**  
To analyze the decision-making processes of the models, the following XAI methods were applied:  
- **Saliency Maps** → Highlights which pixels the model focuses on.  
- **GradientSHAP** → Gradient-based sensitivity analysis measuring input contributions.  
- **Integrated Gradients (IG)** → Attributes importance to features by step-wise tracing of decisions.  

---

## ⚙️ Nasıl Kullanılır? / How to Use  

```bash
# 1. Repo klonlama / Clone repo
git clone https://github.com/username/BuildingFootprint-XAI.git
cd BuildingFootprint-XAI

# 2. Gerekli paketlerin kurulumu / Install requirements
pip install -r requirements.txt

# 3. Eğitim / Training
python train.py --model unetplusplus --data_path ./data --epochs 100

# 4. Tahmin / Inference
python predict.py --model deeplabv3plus --input ./samples/sample.png --output ./results/output.png
