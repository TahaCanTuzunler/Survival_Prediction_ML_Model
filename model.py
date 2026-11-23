# %% 
import pandas as pd
import numpy as np
from IPython.display import display

df = pd.read_excel('Acute_Pancreatitis_Survival_Assignment_Student.xlsx', sheet_name=0) 

print("--- Verisetinin İlk 5 Satırı ---")
display(df.head()) 

print("\n--- Veri Tipleri ve Genel Bilgiler ---")
df.info()

print("\n--- Kolonlardaki Eksik Değer Sayıları ---")
print(df.isnull().sum())

print("\n--- İstatistiksel Özet ---")
display(df.describe())

# %% 
# 2. ADIM: Veriyi Train (Eğitim) ve Test Olarak Bölme (Data Leakage'ı Önleme)
from sklearn.model_selection import train_test_split

# Orijinal verisetinden hedef değişkeni (SURVIVE) ayırıyoruz
X = df.drop('SURVIVE', axis=1) # SURVIVE hariç tüm kolonlar (Bağımsız değişkenler)
y = df['SURVIVE']              # Sadece SURVIVE kolonu (Hedef değişken)

# Veriyi %70 Eğitim (Train) ve %30 Test olarak bölüyoruz
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.30, 
    random_state=42, 
    stratify=y # Ölüm (1) ve Yaşam (0) oranlarının iki sete de dengeli dağılmasını sağlar
)

# Bölme işleminin sonucunu kontrol edelim
print("--- Veri Setlerinin Boyutları ---")
print(f"Orijinal Veri Boyutu: {df.shape}")
print(f"X_train (Eğitim) Boyutu: {X_train.shape}")
print(f"X_test (Test) Boyutu: {X_test.shape}")

# Hedef değişkenin dağılım oranlarını kontrol edelim
print("\n--- Eğitim Setindeki Ölüm/Yaşam Oranı ---")
print(y_train.value_counts(normalize=True))

print("\n--- Test Setindeki Ölüm/Yaşam Oranı ---")
print(y_test.value_counts(normalize=True))

# %%
# 3. ADIM: BUN Kolonu İçin Korelasyon Analizi
# Sadece sayısal (numeric) kolonları seçip korelasyon matrisini hesaplıyoruz
numeric_cols = X_train.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_cols.corr()

print("--- BUN Kolonunun Diğer Değişkenlerle İlişkisi ---")
# BUN kolonuna ait değerleri çekip, büyükten küçüğe sıralıyoruz
bun_corr = correlation_matrix['BUN_MG_DL'].sort_values(ascending=False)
print(bun_corr)

# %%
# 4. ADIM: BUN Kolonunu KNN ile Doldurma
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Sadece sayısal kolonları alıyoruz (Kategorikler KNN'in mesafe hesabını bozar)
numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns

# 1. Ölçeklendirme (Scaling): Sayıları 0-1 arasına çekiyoruz
scaler = MinMaxScaler()
# Sadece X_train'den öğreniyoruz (fit) ve dönüştürüyoruz (transform)
X_train_scaled = scaler.fit_transform(X_train[numeric_cols])
# Öğrenilen bu kuralı X_test'e sadece uyguluyoruz (transform) - Sızıntı engellendi!
X_test_scaled = scaler.transform(X_test[numeric_cols])

# 2. KNN ile Eksik Verileri Doldurma
knn = KNNImputer(n_neighbors=5)
X_train_imputed = knn.fit_transform(X_train_scaled)
X_test_imputed = knn.transform(X_test_scaled)

# 3. Verileri Orijinal Ölçeğine (Gerçek Laboratuvar Sonuçlarına) Geri Döndürme
X_train_filled = scaler.inverse_transform(X_train_imputed)
X_test_filled = scaler.inverse_transform(X_test_imputed)

# 4. Doldurulan verileri sadece BUN kolonu için DataFrame'e geri yazma
X_train['BUN_MG_DL'] = pd.DataFrame(X_train_filled, columns=numeric_cols, index=X_train.index)['BUN_MG_DL']
X_test['BUN_MG_DL'] = pd.DataFrame(X_test_filled, columns=numeric_cols, index=X_test.index)['BUN_MG_DL']

# Sonuç Kontrolü
print("--- Doldurma Sonrası BUN Kolonundaki Boşluk Sayısı ---")
print("Eğitim (Train) Seti BUN Boşluk:", X_train['BUN_MG_DL'].isnull().sum())
print("Test Seti BUN Boşluk:", X_test['BUN_MG_DL'].isnull().sum())

# %%
# 5. ADIM: CRP Kolonu İçin Korelasyon Analizi
# Sayısal kolonların korelasyon matrisini tekrar alıyoruz
numeric_cols = X_train.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_cols.corr()

print("--- CRP Kolonunun Diğer Değişkenlerle İlişkisi ---")
# Bu sefer CRP_MG_DL kolonuna ait değerleri çekip sıralıyoruz
crp_corr = correlation_matrix['CRP_MG_DL'].sort_values(ascending=False)
print(crp_corr)

# %%
# 6. ADIM: CRP Kolonunu Medyan (Ortanca) ile Doldurma
from sklearn.impute import SimpleImputer

# Sadece medyan hesaplayacak bir imputer nesnesi oluşturuyoruz
imputer_crp = SimpleImputer(strategy='median')

# Kural: Sadece X_train'in medyanını öğren (fit) ve X_train'i doldur (transform)
# Not: Scikit-learn çift köşeli parantez (DataFrame formatı) bekler
X_train['CRP_MG_DL'] = imputer_crp.fit_transform(X_train[['CRP_MG_DL']])

# Öğrendiğin o medyan değerini test setine sadece uygula (transform)
X_test['CRP_MG_DL'] = imputer_crp.transform(X_test[['CRP_MG_DL']])

# Sonuç Kontrolü
print("--- Doldurma Sonrası CRP Kolonundaki Boşluk Sayısı ---")
print("Eğitim (Train) Seti CRP Boşluk:", X_train['CRP_MG_DL'].isnull().sum())
print("Test Seti CRP Boşluk:", X_test['CRP_MG_DL'].isnull().sum())

# %%
# 7. ADIM: DIABETES Kolonunu Lojistik Regresyon ile Doldurma
from sklearn.linear_model import LogisticRegression

# 1. Tahmin için kullanılacak özellikleri (features) belirliyoruz.
# Önceden hazırlayıp eksiklerini kapattığımız sayısal kolonları kullanacağız.
# Kendi kendini tahmin etmemesi için DIABETES'i bu listeden çıkarıyoruz.
features = [col for col in numeric_cols if col != 'DIABETES']

# 2. X_train'i, DIABETES kolonu 'Dolu' ve 'Boş' olanlar olarak ikiye ayırıyoruz
train_known = X_train[X_train['DIABETES'].notnull()]
train_unknown = X_train[X_train['DIABETES'].isnull()]

# 3. Lojistik Regresyon modelini kuruyoruz
log_reg = LogisticRegression(max_iter=1000, random_state=42)

# 4. Modeli SADECE X_train'deki 'Dolu' (bilinen) hastalar üzerinde eğitiyoruz (fit)
log_reg.fit(train_known[features], train_known['DIABETES'])

# 5. Model, X_train'deki 'Boş' kısımları tahmin ediyor ve yerlerine yazıyor (transform)
if len(train_unknown) > 0:
    X_train.loc[X_train['DIABETES'].isnull(), 'DIABETES'] = log_reg.predict(train_unknown[features])

# 6. Sızıntı Kuralı: Aynı model, X_test'teki 'Boş' kısımları testin kendi içindeki 
# diğer kolonlara bakarak tahmin ediyor (transform)
test_unknown = X_test[X_test['DIABETES'].isnull()]
if len(test_unknown) > 0:
    X_test.loc[X_test['DIABETES'].isnull(), 'DIABETES'] = log_reg.predict(test_unknown[features])

# Sonuç Kontrolü
print("--- Doldurma Sonrası DIABETES Kolonundaki Boşluk Sayısı ---")
print("Eğitim (Train) Seti DIABETES Boşluk:", X_train['DIABETES'].isnull().sum())
print("Test Seti DIABETES Boşluk:", X_test['DIABETES'].isnull().sum())

# Genel Veri Kontrolü (Tüm X_train'de hiç NaN kaldı mı?)
print("\n--- X_train Kalan Toplam Boşluk Sayısı ---")
print(X_train.isnull().sum().sum(), "adet boşluk kaldı.")

# %%
# 8. ADIM: SEX Kolonunu Manuel Eşleme ile Sayısallaştırma
# Cinsiyetleri açıkça 0 ve 1'e atıyoruz
cinsiyet_sozlugu = {'Female': 1, 'Male': 0}

X_train['SEX'] = X_train['SEX'].map(cinsiyet_sozlugu)
X_test['SEX'] = X_test['SEX'].map(cinsiyet_sozlugu)

# Sonuç Kontrolü
print("--- SEX Kolonu Dönüşüm Sonrası İlk 5 Satır ---")
print(X_train['SEX'].head())


# %%
# 9. ADIM: ETIOLOGY Kolonunu Gruplama ve One-Hot Encoding

# 1. Önce kategorilerin dağılımını (hangi hastalıktan kaç tane olduğunu) görelim
print("--- Gruplamadan Önce ETIOLOGY Dağılımı ---")
print(X_train['ETIOLOGY'].value_counts())

# 2. En çok tekrar eden ilk 5 kategorinin isimlerini bir listeye (index) alıyoruz
# SADECE X_train'in popüler kategorilerini öğreniyoruz (Fit mantığı)
top_5_categories = X_train['ETIOLOGY'].value_counts().nlargest(5).index

# 3. İlk 5'in içinde olmayan tüm nadir kategorileri 'Other' (Diğer) olarak değiştiriyoruz
X_train['ETIOLOGY'] = X_train['ETIOLOGY'].apply(lambda x: x if x in top_5_categories else 'Other')
X_test['ETIOLOGY'] = X_test['ETIOLOGY'].apply(lambda x: x if x in top_5_categories else 'Other')

# 4. Sadeleşmiş kategori sayımızla Pandas üzerinden One-Hot Encoding yapıyoruz
# drop_first=True parametresi "Kukla Değişken Tuzağı"nı (Dummy Variable Trap) önler.
X_train = pd.get_dummies(X_train, columns=['ETIOLOGY'], drop_first=True, dtype=int)
X_test = pd.get_dummies(X_test, columns=['ETIOLOGY'], drop_first=True, dtype=int)

# Eğitim ve Test setindeki kolonların tam eşleştiğinden emin olmak için (sürpriz kategorilere karşı):
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# Sonuç Kontrolü
print("\n--- Dönüşüm Sonrası Yeni ETIOLOGY Kolonları ---")
print(X_train.filter(like='ETIOLOGY').head())

# %%
# 10. ADIM: Modellerin Kurulması, Eğitilmesi ve Tahmin (Survival Prediction)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# 1. Modelleri tanımlıyoruz
# Lojistik regresyonun iterasyon limitini yüksek tutuyoruz ki veriyi tam öğrenebilsin
log_model = LogisticRegression(max_iter=2000, random_state=42)
gnb_model = GaussianNB()

# 2. Modelleri SADECE eğitim setiyle eğitiyoruz (Öğrenme aşaması)
log_model.fit(X_train, y_train)
gnb_model.fit(X_train, y_train)

# 3. Eğitilen modellere, daha önce hiç görmedikleri X_test verisini verip sınav yapıyoruz
# DİKKAT: y_test (gerçek cevaplar) modellere asla verilmez!
log_pred = log_model.predict(X_test)
gnb_pred = gnb_model.predict(X_test)

# 4. ROC-AUC metriği için hastaların sadece 0 veya 1 olma durumlarını değil, 
# 'yüzde kaç ihtimalle' 1 (Vefat) sınıfına girdiklerini (olasılıklarını) de alıyoruz
log_pred_proba = log_model.predict_proba(X_test)[:, 1]
gnb_pred_proba = gnb_model.predict_proba(X_test)[:, 1]

print("Modeller başarıyla eğitildi ve test seti üzerinde tahminler yapıldı!")


# %%
# 11. ADIM: Modellerin Değerlendirilmesi (Evaluation ve Karşılaştırma)
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

def raporla(model_adi, gercek_y, tahmin_y, tahmin_y_proba):
    print(f"\n{'='*50}")
    print(f"      MODEL: {model_adi}")
    print(f"{'='*50}")
    
    # 1. Karmaşıklık Matrisi (Confusion Matrix)
    cm = confusion_matrix(gercek_y, tahmin_y)
    print("\n--- Karmaşıklık Matrisi ---")
    print(f"Gerçekte Yaşayanı 'Yaşayacak' Bildi (True Negative)  : {cm[0][0]}")
    print(f"Gerçekte Yaşayanı 'Ölecek' Bildi (False Positive)    : {cm[0][1]} (Yanlış Alarm)")
    print(f"Gerçekte Öleni 'Yaşayacak' Bildi (False Negative)    : {cm[1][0]} (TEHLİKELİ KAÇAK)")
    print(f"Gerçekte Öleni 'Ölecek' Bildi (True Positive)        : {cm[1][1]} (Başarılı Yakalama)")
    
    # 2. ROC-AUC Skoru (Modelin 0 ve 1'i birbirinden ayırma gücü)
    roc_auc = roc_auc_score(gercek_y, tahmin_y_proba)
    print(f"\n--- ROC-AUC Skoru ---")
    print(f"Skor: {roc_auc:.4f} (1.0'a ne kadar yakınsa o kadar iyi)")
    
    # 3. Sınıflandırma Raporu (Accuracy, Precision, Recall, F1)
    print("\n--- Sınıflandırma Raporu ---")
    print(classification_report(gercek_y, tahmin_y))

# Lojistik Regresyon için sonuçları yazdır
raporla("Lojistik Regresyon", y_test, log_pred, log_pred_proba)

# Gaussian Naive Bayes için sonuçları yazdır
raporla("Gaussian Naive Bayes", y_test, gnb_pred, gnb_pred_proba)

# %%
