# 🏙️ BuildingFootprint-XAI  
*Açıklanabilir Yapay Zekâ ile Çok Yüksek Çözünürlüklü Görüntülerden Bina Çıkarımı*  

---

## 📑 İçindekiler / Table of Contents  
- [🚀 Kısa Özet / Summary](#-kısa-özet--summary)   
- [🛰️ Kullanılan Veri / Data Used](#-kullanılan-veri--data-used)
- [🧠 Modeller / Models](#-modeller--models)
- [🔎 XAI Yöntemleri / XAI-Methods](#-xai-yöntemleri--xai-methods)
- [⚙️ Nasıl Kullanılır / How to Use](#-nasıl-kullanılır--how-to-use)
- [📊 Sonuçlar / Results](#-sonuçlar--results)


---

## 🚀 Kısa Özet / Summary  

🇹🇷 **Türkçe:**  
Bu proje, TÜBİTAK 1002 desteğiyle yürütülmüş olup, çok yüksek çözünürlüklü uydu ve hava görüntülerinden bina ayak izlerinin otomatik çıkarımı üzerine odaklanmaktadır. Çalışmada **U-Net++**, **DeepLabV3+**, **PSPNet** ve **DeepSwinLite** gibi güncel derin öğrenme modelleri uygulanmış; ayrıca açıklanabilir yapay zekâ (XAI) yöntemleri kullanılarak bu modellerin karar mekanizmaları incelenmiştir. Elde edilen sonuçlar, farklı kentsel dokularda performans farklılıklarını ortaya koymakta ve yöntemlerin **şehir planlama, afet yönetimi ve sürdürülebilir şehircilik** uygulamalarında güvenilir bir şekilde kullanılabileceğini göstermektedir.  

🇬🇧 **English:**  
This project, supported by TÜBİTAK 1002, focuses on the automatic extraction of building footprints from very high-resolution satellite and aerial imagery. State-of-the-art deep learning models such as **U-Net++**, **DeepLabV3+**, **PSPNet**, and **DeepSwinLite** were applied, and explainable artificial intelligence (XAI) techniques were employed to analyze the decision-making processes of these models. The results reveal performance differences across various urban patterns and demonstrate that the proposed approaches can be reliably used in applications such as **urban planning, disaster management, and sustainable city development**.  

---

## 🛰️ Kullanılan Veri / Data Used  

🇹🇷 **Türkçe:**  
Projede iki farklı veri seti kullanılmıştır:  

- **SPOT 6/7 (Fransa, Pyrénées-Orientales)**  
  - Sağlayıcı: *Airbus / IGN* → [https://space-solutions.airbus.com/imagery/sample-imagery/spot-2-ortho-pyrenees-orientales-france-july-2023/](https://space-solutions.airbus.com/imagery/sample-imagery/spot-2-ortho-pyrenees-orientales-france-july-2023/)  
  - Mekânsal çözünürlük: 1.5 m (pansharpened ≈ 0.5 m)  
  - Alım yılı: 2016  
  - Bina sınırları shapefile: *IGN* → [https://thisme.cines.teledetection.fr/home](https://thisme.cines.teledetection.fr/home)  

- **MAXAR (Türkiye, İzmir)**  
  - Sağlayıcı: *Maxar Technologies* → [https://www.maxar.com/open-data/turkey-earthquake](https://www.maxar.com/open-data/turkey-earthquake)  
  - Mekânsal çözünürlük: 0.3 m  
  - Alım yılı: 2020  
  - Bina sınırları shapefile: *HERE Maps* → [https://www.here.com/developer/sample-map-data](https://www.here.com/developer/sample-map-data)  

> 📌 **Not:** Görüntüler ve shapefile verileri lisans kısıtları nedeniyle bu repoda paylaşılamamaktadır. Araştırmacılar ilgili sağlayıcılardan resmi olarak temin edebilirler.

---

🇬🇧 **English:**  
Two datasets were used in this project:  

- **SPOT 6/7 (France, Pyrénées-Orientales)**  
  - Provider: *Airbus / IGN* → [https://space-solutions.airbus.com/imagery/sample-imagery/spot-2-ortho-pyrenees-orientales-france-july-2023/](https://space-solutions.airbus.com/imagery/sample-imagery/spot-2-ortho-pyrenees-orientales-france-july-2023/)  
  - Spatial resolution: 1.5 m (pansharpened ≈ 0.5 m)  
  - Acquisition year: 2016  
  - Building footprints shapefile: *IGN* → [https://thisme.cines.teledetection.fr/home](https://thisme.cines.teledetection.fr/home)    

- **MAXAR (Turkey, Izmir)**  
  - Provider: *Maxar Technologies* → [https://www.maxar.com/open-data/turkey-earthquake](https://www.maxar.com/open-data/turkey-earthquake)  
  - Spatial resolution: 0.3 m  
  - Acquisition year: 2020  
  - Building footprints shapefile: *HERE Maps* → [https://www.here.com/developer/sample-map-data](https://www.here.com/developer/sample-map-data)  

> 📌 **Note:** Due to licensing restrictions, imagery and shapefile data are not shared in this repository. Researchers can obtain them from the providers. For reproducibility, codes are also compatible with open datasets such as [Massachusetts Buildings](https://www.cs.toronto.edu/~vmnih/data/) and [WHU Building Dataset](http://gpcv.whu.edu.cn/data/).  


---

## 🧠 Modeller / Models  

| Model          | Açıklama (TR) | Description (EN) |
|----------------|---------------|------------------|
| **U-Net++**    | Gelişmiş U-Net, skip-connection yapısıyla daha hassas segmentasyon sağlar. | Enhanced U-Net with nested skip connections for more accurate segmentation. |
| **DeepLabV3+** | Atrous konvolüsyon + ASPP ile çok ölçekli öğrenme sağlar. | Employs atrous convolution and Atrous Spatial Pyramid Pooling (ASPP) for multi-scale learning. |
| **PSPNet**     | Pyramid Pooling ile bağlamsal bilgi yakalar. | Utilizes Pyramid Pooling Module to capture contextual information. |


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

## ⚙️ Nasıl Kullanılır / How to Use  

> ⏳ **Not / Note:** Bu proje kapsamında kodlar ve detaylı yönergeler **yakında paylaşılacaktır**.  
> ⏳ **Note:** Codes and detailed instructions will be **coming soon** in this repository.  


---

## 📊 Sonuçlar / Results  

Örnek doğruluk metrikleri ve görselleştirmeler burada paylaşılacaktır.  
Example accuracy metrics and visualizations will be shared here.  

> 📌 **Not / Note:** Tüm kodlar ve ayrıntılı sonuçlar, gerekli düzenlemeler tamamlandıktan ve ilgili yayın süreci sonuçlandıktan sonra bu depoya eklenecektir.  
> 📌 **Note:** All codes and detailed results will be made available in this repository once the necessary refinements are completed and the related publication process is finalized.  
