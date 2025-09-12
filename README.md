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
Bu proje, TÜBİTAK 1002 desteğiyle yürütülmüş olup, çok yüksek çözünürlüklü uydu ve hava görüntülerinden bina ayak izlerinin otomatik çıkarımı üzerine odaklanmaktadır. Çalışmada **U-Net++**, **DeepLabV3+** ve **PSPNet** gibi güncel derin öğrenme modelleri uygulanmış; ayrıca açıklanabilir yapay zekâ (XAI) yöntemleri kullanılarak bu modellerin karar mekanizmaları incelenmiştir. Elde edilen sonuçlar, farklı kentsel dokularda performans farklılıklarını ortaya koymakta ve yöntemlerin **şehir planlama, afet yönetimi ve sürdürülebilir şehircilik** uygulamalarında güvenilir bir şekilde kullanılabileceğini göstermektedir.  

🇬🇧 **English:**  
This project, supported by TÜBİTAK 1002, focuses on the automatic extraction of building footprints from very high-resolution satellite and aerial imagery. State-of-the-art deep learning models such as **U-Net++**, **DeepLabV3+** and **PSPNet** were applied, and explainable artificial intelligence (XAI) techniques were employed to analyze the decision-making processes of these models. The results reveal performance differences across various urban patterns and demonstrate that the proposed approaches can be reliably used in applications such as **urban planning, disaster management, and sustainable city development**.  

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
 
- **Ön-işleme adımları:**

  - Ortofoto ve shapefile verilerinin projeksiyon dönüşümü ve hizalanması
  - Bina shapefile’larının raster maske formatına dönüştürülmesi
  - Görüntülerin 512×512/256x256 yamalara bölünmesi (sliding window)
  - Eğitim, doğrulama ve test için ayrı listeler (.txt) ile veri bölme
  - Normalizasyon ve veri artırma (rotasyon, çevirme vb.) işlemleri

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

- **Preprocessing steps:**

  - Projection alignment between imagery and shapefiles
  - Conversion of building shapefiles into raster masks
  - Splitting images into 512×512 patches (sliding window)
  - Creating train/validation/test splits with .txt lists
  - Applying normalization and data augmentation (rotation, flipping, etc.)

> 📌 **Note:** Due to licensing restrictions, imagery and shapefile data are not shared in this repository. Researchers can obtain them from the providers.


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
- **Integrated Gradients** → Model kararını adım adım izleyerek özelliklerin önemini belirler.  

🇬🇧 **English:**  
To analyze the decision-making processes of the models, the following XAI methods were applied:  
- **Saliency Maps** → Highlights which pixels the model focuses on.  
- **GradientSHAP** → Gradient-based sensitivity analysis measuring input contributions.  
- **Integrated Gradients** → Attributes importance to features by step-wise tracing of decisions.  

---

## ⚙️ Nasıl Kullanılır / How to Use  

🇹🇷 Türkçe:

- train.py dosyası ile modeller (U-Net++, DeepLabv3+, PSPNet) SPOT6/7 veya MAXAR_İzmir veri setlerinde eğitilir.
- Eğitim sonucunda en iyi ağırlık dosyası ve özet metrikler kaydedilir.
- infer_xai.py dosyası tek bir görsel veya klasör için tahmin ve XAI haritaları üretir.
- eval_xai_metrics.py dosyası ile 3 XAI yönteminin (Saliency, IntegratedGradients, GradientShap) 10 farklı metriği hesaplanır.

🇬🇧 English:

- Models (U-Net++, DeepLabv3+, PSPNet) are trained on SPOT6/7 or MAXAR_İzmir datasets using train.py.
- Training outputs include the best model weights and a summary of validation metrics.
- infer_xai.py generates predictions and attribution maps for a single image or a folder.
- eval_xai_metrics.py computes 10 different metrics for the 3 XAI methods (Saliency, IntegratedGradients, GradientShap).

---

## 📊 Sonuçlar / Results  

🇹🇷 **Türkçe:**

- Çalışmada iki farklı veri seti (SPOT6/7 ve MAXAR_İzmir) üzerinde bina çıkarımı yapılmıştır.
- Üç derin öğrenme modeli (U-Net++, DeepLabv3+, PSPNet) karşılaştırılmış, doğruluk metrikleri olarak mIoU, Dice, Recall, Precision ve Accuracy raporlanmıştır.
- Ayrıca üç farklı XAI yöntemi (Saliency, IntegratedGradients, GradientShap) uygulanmış ve her yöntem için 10 açıklanabilirlik metriği (Continuity, FaithfulnessEstimate, AUC, Sparseness, Complexity, RRA, RMA, FaithfulnessCorr, Infidelity, MPRT) hesaplanmıştır.
- Bu değerlendirmeler, hem modellerin doğruluğunu hem de açıklanabilirlik kalitesini birlikte göstermektedir.
- Örnek tahmin görselleri, XAI haritaları ve ayrıntılı tablolar yayın süreci tamamlandıktan sonra bu depoda paylaşılacaktır.

🇬🇧 **English:**  

- Building extraction experiments were conducted on two datasets: SPOT6/7 and MAXAR_İzmir.
- Three deep learning models (U-Net++, DeepLabv3+, PSPNet) were compared, and validation metrics including mIoU, Dice, Recall, Precision, and Accuracy were reported.
- In addition, three XAI methods (Saliency, IntegratedGradients, GradientShap) were applied, and for each method 10 explainability metrics (Continuity, FaithfulnessEstimate, AUC, Sparseness, Complexity, RRA, RMA, FaithfulnessCorr, Infidelity, MPRT) were computed.
- These evaluations demonstrate both the segmentation accuracy and the quality of the model explanations.
- Example predictions, attribution maps, and detailed quantitative results will be shared in this repository after the publication process is finalized.



