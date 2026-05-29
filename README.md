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
├── docs/
├── notebooks/
├── reports/
├── src/
├── visuals/
└── README.md
```


## Veri Kaynağı
Veri seti [Kaggle - Football Data from Transfermarkt](https://www.kaggle.com/datasets/davidcariboo/player-scores) platformundan 5 Nisan 2026 tarihinde elde edilmiştir.

Kullanılan tablolar:
- `players.csv` — oyuncu demografik bilgileri
- `player_valuations.csv` — piyasa değeri geçmişi (en güncel değer)
- `appearances.csv` — maç bazlı performans istatistikleri
- `competitions.csv` — lig bilgileri
- `clubs.csv` — kulüp bilgileri


## Ekip
| İsim                |
|---------------------|
| Yavuz Selim Çoraklı |
| Miran Emre Eser     |
| İrem Çelebi         |
