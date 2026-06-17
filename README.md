# Futbolcu Piyasa Analizi
Futbol dünyasındaki oyuncu değerlerini etkileyen faktörler uçtan uca bir süreçle incelenmektedir.


## Problem
Mevcut transfer piyasasında oyuncu değerlemeleri genellikle veri yerine öznel gözlemlere dayanmakta, bu da kulüpler için ciddi finansal belirsizlikler oluşturmaktadır.


## Amaç ve Kapsam
Futbolcuların saha içi performans istatistikleri (gol, asist vb.), demografik verileri (yaş, boy vb.) ve bulundukları ekosistemin (lig vb.)
ekonomik değerleri üzerindeki etkisini nicel yöntemlerle ortaya koymaktır. Çalışma, spor endüstrisinde veri odaklı karar alma süreçlerini desteklemek amacıyla kurgulanmıştır.


## Hedeflenen Çıktılar
Elde edilmesi beklenen çıktılar aşağıda verilmiştir:
- Veri Analizi ve Görselleştirme: Oyuncu özellikleri ile piyasa değeri arasındaki ilişkileri açıklayan yorumlanmış grafikler.
- Tahmin Modeli: Linear Regression (Doğrusal Regresyon) ve Random Forest (Rastgele Orman) algoritmaları kullanılarak geliştirilen genel ve mevki bazlı piyasa değeri tahmin modelleri.
- Performans Raporu: Modellerin başarısının RMSE gibi teknik metriklerle karşılaştırıldığı ve zayıf/güçlü yönlerin analiz edildiği dökümantasyon.
- Uçtan Uca Depo: Veri toplama aşamasından model sonucuna kadar tüm sürecin şeffaf bir şekilde izlenebildiği, düzenli bir GitHub deposu.


## Dosya Yapısı
```
futbolcu-piyasa-analizi/
├── data/
├── models/
├── notebooks/
├── src/
├── visuals/
├── requirements.txt
└── README.md
```


## Kurulum ve Çalıştırma
1. Veri setini [Kaggle - Football Data from Transfermarkt](https://www.kaggle.com/datasets/davidcariboo/player-scores) adresinden indirip `data/raw/` klasörüne yerleştirin.
2. Bağımlılıkları yükleyin: `pip install -r requirements.txt`
3. Keşifsel analizi çalıştırın: `notebooks/01_eda.ipynb`
4. Ön işleme adımlarını çalıştırın: `notebooks/02_preprocessing.ipynb`
5. Modelleme adımlarını çalıştırın: `notebooks/03_modeling.ipynb`
6. Modeli eğitip kaydedin: `python src/build.py`
7. Arayüzü başlatın: `streamlit run src/app.py`


## Veri Kaynağı
Veri seti [Kaggle - Football Data from Transfermarkt](https://www.kaggle.com/datasets/davidcariboo/player-scores) platformundan elde edilmiştir.

Kullanılan tablolar:
- `players.csv` — oyuncu demografik bilgileri
- `player_valuations.csv` — piyasa değeri geçmişi (en güncel değer)
- `appearances.csv` — maç bazlı performans istatistikleri
- `competitions.csv` — lig bilgileri
- `clubs.csv` — kulüp bilgileri


## Ekip
| İsim                   | Görev Dağılımı         |
|------------------------|------------------------|
| Yavuz Selim Çoraklı    | Veri Analizi ve İşleme |
| Miran Emre Eser        | Modelleme ve Arayüz    |
| İrem Çelebi            | Modelleme ve Rapor     |