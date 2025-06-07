import mysql.connector
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
import warnings
warnings.filterwarnings('ignore')
DB_USERNAME = 'root'
DB_PASSWORD = 'Ludfyrahman123'
DB_HOST = 'localhost'
DB_NAME = 'skripsi_TI_sentimen'
# Download NLTK data (jalankan sekali saja)
try:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')

except:
    pass

class SentimentClassifier():
    def __init__(self):
        """
        Initialize classifier dengan konfigurasi database
        
        db_config: dict dengan keys: host, user, password, database
        """
        db_config = {
            'host': 'localhost',
            'user': DB_USERNAME,
            'password': DB_PASSWORD,
            'database': DB_NAME
        }
        self.db_config = db_config
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.knn_model = KNeighborsClassifier(n_neighbors=5)
        self.lda_model = LinearDiscriminantAnalysis()
        self.label_encoder = LabelEncoder()
        self.stemmer = PorterStemmer()
        
        # Stop words untuk bahasa Indonesia dan Inggris
        self.stop_words = set(stopwords.words('indonesian') + stopwords.words('english'))
    
    def connect_db(self):
        """Koneksi ke database MySQL"""
        try:
            connection = mysql.connector.connect(**self.db_config)
            return connection
        except mysql.connector.Error as e:
            print(f"Error connecting to MySQL: {e}")
            return None
    
    def load_data(self):
        """Load data dari database MySQL"""
        connection = self.connect_db()
        if connection is None:
            return None
        
        try:
            query = "SELECT description, category FROM training"
            df = pd.read_sql(query, connection)
            connection.close()
            
            print(f"Data berhasil dimuat: {len(df)} baris")
            print(f"Distribusi kategori:")
            print(df['category'].value_counts())
            
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def preprocess_text(self, text):
        """Preprocessing teks"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions dan hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove punctuation dan angka
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stopwords dan stemming
        tokens = [self.stemmer.stem(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def prepare_data(self, df):
        """Persiapan data untuk training"""
        # Preprocessing teks
        print("Preprocessing teks...")
        df['processed_text'] = df['description'].apply(self.preprocess_text)
        
        # Remove empty texts
        df = df[df['processed_text'].str.len() > 0]
        
        # Encode labels
        y = self.label_encoder.fit_transform(df['category'])
        
        # Vectorization
        print("Vectorizing teks...")
        X = self.vectorizer.fit_transform(df['processed_text'])
        
        return X, y, df
    
    def train_models(self, X, y):
        """Training model KNN dan LDA"""
        print("Training models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Training KNN
        print("Training KNN...")
        self.knn_model.fit(X_train, y_train)
        
        # Training LDA
        print("Training LDA...")
        # Convert sparse matrix to dense untuk LDA
        X_train_dense = X_train.toarray()
        X_test_dense = X_test.toarray()
        
        self.lda_model.fit(X_train_dense, y_train)
        
        return X_train, X_test, y_train, y_test, X_train_dense, X_test_dense
    
    def evaluate_models(self, X_test, y_test, X_test_dense):
        """Evaluasi performa model"""
        print("\n=== EVALUASI MODEL ===")
        
        # Prediksi KNN
        knn_pred = self.knn_model.predict(X_test)
        knn_accuracy = accuracy_score(y_test, knn_pred)
        
        # Prediksi LDA
        lda_pred = self.lda_model.predict(X_test_dense)
        lda_accuracy = accuracy_score(y_test, lda_pred)
        
        print(f"\nKNN Accuracy: {knn_accuracy:.4f}")
        print(f"LDA Accuracy: {lda_accuracy:.4f}")
        
        # Classification report untuk KNN
        print("\n=== KNN Classification Report ===")
        print(classification_report(y_test, knn_pred, 
                                  target_names=self.label_encoder.classes_))
        
        # Classification report untuk LDA
        print("\n=== LDA Classification Report ===")
        print(classification_report(y_test, lda_pred, 
                                  target_names=self.label_encoder.classes_))
        
        # Confusion Matrix
        print("\n=== Confusion Matrix KNN ===")
        print(confusion_matrix(y_test, knn_pred))
        
        print("\n=== Confusion Matrix LDA ===")
        print(confusion_matrix(y_test, lda_pred))
        
        return knn_accuracy, lda_accuracy
    
    def predict_sentiment(self, text, model_type='knn'):
        """Prediksi sentimen untuk teks baru"""
        # Preprocessing
        processed_text = self.preprocess_text(text)
        
        # Vectorization
        text_vector = self.vectorizer.transform([processed_text])
        
        # Prediksi
        if model_type.lower() == 'knn':
            prediction = self.knn_model.predict(text_vector)
            probability = self.knn_model.predict_proba(text_vector)
        else:  # LDA
            text_vector_dense = text_vector.toarray()
            prediction = self.lda_model.predict(text_vector_dense)
            probability = self.lda_model.predict_proba(text_vector_dense)
        
        # Decode label
        predicted_label = self.label_encoder.inverse_transform(prediction)[0]
        
        return predicted_label, probability[0]
    
    def cross_validation(self, X, y):
        """Cross validation untuk evaluasi yang lebih robust"""
        print("\n=== CROSS VALIDATION ===")
        
        # KNN Cross Validation
        knn_scores = cross_val_score(self.knn_model, X, y, cv=5, scoring='accuracy')
        print(f"KNN CV Accuracy: {knn_scores.mean():.4f} (+/- {knn_scores.std() * 2:.4f})")
        
        # LDA Cross Validation
        X_dense = X.toarray()
        lda_scores = cross_val_score(self.lda_model, X_dense, y, cv=5, scoring='accuracy')
        print(f"LDA CV Accuracy: {lda_scores.mean():.4f} (+/- {lda_scores.std() * 2:.4f})")
        
        return knn_scores, lda_scores
    def classify(self, text):
        """Fungsi untuk mengklasifikasikan teks"""
        # Preprocessing
        df = self.load_data()
        if df is None:
            print("Gagal memuat data!")
            return
        
        X, y, _ = self.prepare_data(df)
        self.train_models(X, y)

        knn_pred, knn_prob = self.predict_sentiment(text, 'knn')
        lda_pred, lda_prob = self.predict_sentiment(text, 'lda')
        return knn_pred;
def main():
    # Konfigurasi database
    db_config = {
        'host': 'localhost',
        'user': DB_USERNAME,
        'password': DB_PASSWORD,
        'database': DB_NAME
    }
    
    # Initialize classifier
    classifier = SentimentClassifier(db_config)
    
    # Load data
    print("Loading data dari database...")
    df = classifier.load_data()
    if df is None:
        print("Gagal memuat data!")
        return
    
    # Prepare data
    X, y, processed_df = classifier.prepare_data(df)
    
    # Train models
    X_train, X_test, y_train, y_test, X_train_dense, X_test_dense = classifier.train_models(X, y)
    
    # Evaluate models
    knn_acc, lda_acc = classifier.evaluate_models(X_test, y_test, X_test_dense)
    
    # Cross validation
    classifier.cross_validation(X, y)
    
    # Contoh prediksi
    print("\n=== CONTOH PREDIKSI ===")
    test_texts = [
        "Produk ini sangat bagus dan memuaskan",
        "Pelayanan buruk sekali, sangat mengecewakan",
        "Produk standar, tidak istimewa tapi juga tidak buruk"
    ]
    
    for text in test_texts:
        knn_pred, knn_prob = classifier.predict_sentiment(text, 'knn')
        lda_pred, lda_prob = classifier.predict_sentiment(text, 'lda')
        
        print(f"\nTeks: '{text}'")
        print(f"KNN Prediksi: {knn_pred} (confidence: {max(knn_prob):.4f})")
        print(f"LDA Prediksi: {lda_pred} (confidence: {max(lda_prob):.4f})")

if __name__ == "__main__":
    main()

# Fungsi tambahan untuk prediksi interaktif
def interactive_prediction():
    """Fungsi untuk prediksi interaktif"""
    
    
    classifier = SentimentClassifier()
    
    # Load dan train model (dalam implementasi nyata, simpan model yang sudah trained)
    df = classifier.load_data()
    if df is None:
        print("Gagal memuat data!")
        return
    
    X, y, _ = classifier.prepare_data(df)
    classifier.train_models(X, y)
    
    print("=== PREDIKSI SENTIMEN INTERAKTIF ===")
    print("Ketik 'quit' untuk keluar")
    
    while True:
        text = input("\nMasukkan teks untuk klasifikasi: ")
        if text.lower() == 'quit':
            break
        
        knn_pred, knn_prob = classifier.predict_sentiment(text, 'knn')
        lda_pred, lda_prob = classifier.predict_sentiment(text, 'lda')
        
        print(f"KNN Prediksi: {knn_pred} (confidence: {max(knn_prob):.4f})")
        print(f"LDA Prediksi: {lda_pred} (confidence: {max(lda_prob):.4f})")
def checkCategory(text):
    """Fungsi untuk memeriksa kategori sentimen dari teks"""
    classifier = SentimentClassifier()
    
    # Load dan train model (dalam implementasi nyata, simpan model yang sudah trained)
    df = classifier.load_data()
    if df is None:
        print("Gagal memuat data!")
        return
    
    X, y, _ = classifier.prepare_data(df)
    classifier.train_models(X, y)

    knn_pred, knn_prob = classifier.predict_sentiment(text, 'knn')
    lda_pred, lda_prob = classifier.predict_sentiment(text, 'lda')

    return knn_pred;

# Uncomment baris di bawah untuk menjalankan prediksi interaktif
# interactive_prediction()