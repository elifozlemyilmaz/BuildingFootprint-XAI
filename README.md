# BuildingFootprint-XAI

Açıklanabilir Yapay Zekâ ile Çok Yüksek Çözünürlüklü Görüntülerden Bina Çıkarımı

Bu proje, TÜBİTAK 1002 desteğiyle yürütülmüş olup, çok yüksek çözünürlüklü uydu ve hava görüntülerinden bina ayak izlerinin otomatik çıkarımı üzerine odaklanmaktadır. Çalışmada U-Net++, DeepLabV3+, PSPNet ve DeepSwinLite gibi güncel derin öğrenme modelleri uygulanmış; ayrıca açıklanabilir yapay zekâ (XAI) yöntemleri kullanılarak bu modellerin karar mekanizmaları incelenmiştir. Elde edilen sonuçlar, farklı kentsel dokularda performans farklılıklarını ortaya koymakta ve yöntemlerin şehir planlama, afet yönetimi ve sürdürülebilir şehircilik uygulamalarında güvenilir bir şekilde kullanılabileceğini göstermektedir.

Projede iki farklı veri seti kullanılmıştır: SPOT 6/7 (Fransa, Pyrénées-Orientales) ve MAXAR (Türkiye, İzmir). Bu veriler çok yüksek çözünürlüklü olup farklı kentsel dokuları temsil etmektedir. Görüntülerin yanı sıra, bina sınırlarını içeren shapefile (vektör) verileri de kullanılmıştır:
Fransa / Pyrénées-Orientales bölgesine ait bina sınırları shapefile’ı, Institut National de l'Information Géographique et Forestière (IGN) tarafından sağlanmaktadır → https://geoservices.ign.fr/
Türkiye / İzmir için kullanılan bina sınırları shapefile verileri, HERE Maps veritabanından elde edilmiştir → https://developer.here.com/
📌 Lisans kısıtları nedeniyle ham görüntüler ve shapefile verileri bu repoda paylaşılamamakta, ancak araştırmacılar ilgili sağlayıcılardan resmi olarak temin edebilirler.


XAI-Based Building Footprint Extraction from VHR Imagery 

This project, supported by TÜBİTAK 1002, focuses on the automatic extraction of building footprints from very high-resolution satellite and aerial imagery. State-of-the-art deep learning models such as U-Net++, DeepLabV3+, PSPNet, and DeepSwinLite were applied, and explainable artificial intelligence (XAI) techniques were employed to analyze the decision-making processes of these models. The results reveal performance differences across various urban patterns and demonstrate that the proposed approaches can be reliably used in applications such as urban planning, disaster management, and sustainable city development.

Two datasets were utilized in this project: SPOT 6/7 (France, Pyrénées-Orientales) and MAXAR (Turkey, Izmir). These very high-resolution datasets represent diverse urban patterns. In addition to imagery, shapefile (vector) data containing building footprints were also used:
Building footprint shapefiles for France / Pyrénées-Orientales are provided by the Institut National de l'Information Géographique et Forestière (IGN) → https://geoservices.ign.fr/
Building footprint shapefiles for Turkey / Izmir were obtained from the HERE Maps database → https://developer.here.com/
📌 Due to licensing restrictions, the raw imagery and shapefile data cannot be shared in this repository, but researchers can obtain them officially from the respective providers.
