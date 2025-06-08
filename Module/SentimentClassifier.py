import mysql.connector
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.lda_model
from wordcloud import WordCloud
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
import warnings
import plotly.express as px
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle

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

class SentimentClassifier:
    def __init__(self):
        """Initialize analyzer with database configuration and visualization capabilities"""
        db_config = {
            'host': DB_HOST,
            'user': DB_USERNAME,
            'password': DB_PASSWORD,
            'database': DB_NAME
        }
        self.db_config = db_config
        
        # Classification models
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.knn_model = KNeighborsClassifier(n_neighbors=5)
        self.lda_classifier = LinearDiscriminantAnalysis()
        
        # Topic modeling
        self.count_vectorizer = CountVectorizer(max_features=1000, ngram_range=(1, 2))
        self.lda_topic_model = LatentDirichletAllocation(
            n_components=5, 
            random_state=42, 
            max_iter=10,
            learning_method='online'
        )
        
        self.label_encoder = LabelEncoder()
        self.stemmer = PorterStemmer()
        
        # Stop words
        self.stop_words = set(stopwords.words('indonesian') + stopwords.words('english'))
        
        # Store processed data
        self.processed_data = None
        self.topic_model_fitted = False
        
    def connect_db(self):
        """Connect to MySQL database"""
        try:
            connection = mysql.connector.connect(**self.db_config)
            return connection
        except mysql.connector.Error as e:
            print(f"Error connecting to MySQL: {e}")
            return None
    
    def load_data(self):
        """Load data from MySQL database"""
        connection = self.connect_db()
        if connection is None:
            return None
        
        try:
            query = "SELECT description, category FROM training"
            df = pd.read_sql(query, connection)
            connection.close()
            
            print(f"Data loaded successfully: {len(df)} rows")
            print(f"Category distribution:")
            print(df['category'].value_counts())
            
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def preprocess_text(self, text):
        """Text preprocessing"""
        if pd.isna(text):
            return ""
        
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        
        tokens = word_tokenize(text)
        tokens = [self.stemmer.stem(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def prepare_data(self, df):
        """Prepare data for training and topic modeling"""
        print("Preprocessing text...")
        df['processed_text'] = df['description'].apply(self.preprocess_text)
        df = df[df['processed_text'].str.len() > 0]
        
        # Store processed data
        self.processed_data = df
        
        # For classification
        y = self.label_encoder.fit_transform(df['category'])
        X_tfidf = self.tfidf_vectorizer.fit_transform(df['processed_text'])
        
        # For topic modeling
        X_count = self.count_vectorizer.fit_transform(df['processed_text'])
        
        return X_tfidf, X_count, y, df
    
    def train_classification_models(self, X_tfidf, y):
        """Train KNN and LDA classification models"""
        print("Training classification models...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train KNN
        self.knn_model.fit(X_train, y_train)
        
        # Train LDA classifier
        X_train_dense = X_train.toarray()
        X_test_dense = X_test.toarray()
        self.lda_classifier.fit(X_train_dense, y_train)
        
        return X_train, X_test, y_train, y_test, X_train_dense, X_test_dense
    
    def train_topic_model(self, X_count):
        """Train LDA topic model"""
        print("Training topic model...")
        self.lda_topic_model.fit(X_count)
        self.topic_model_fitted = True
        return self.lda_topic_model
    
    def create_ldavis_visualization(self, X_count, save_path='lda_visualization.html'):
        """Create interactive LDAvis visualization"""
        # if not self.topic_model_fitted:
        #     print("Topic model not fitted yet!")
        #     return None
        
        # print("Creating LDAvis visualization...")
        
        # # Prepare LDAvis
        # vis_data = pyLDAvis.lda_model.prepare(
        #     self.lda_topic_model, 
        #     X_count, 
        #     self.count_vectorizer,
        #     mds='tsne'
        # )
        
        # # Save to HTML
        # pyLDAvis.save_html(vis_data, save_path)
        # print(f"LDAvis visualization saved to {save_path}")
        
        # return vis_data
        # make intertopic visualization using gensim
        print("Creating LDAvis visualization...")
        vis_data = pyLDAvis.gensim_models.prepare(
            self.lda_topic_model, 
            X_count, 
            self.count_vectorizer,
            mds='tsne'
        )
        # Save to HTML
        pyLDAvis.save_html(vis_data, save_path)
        print(f"LDAvis visualization saved to {save_path}")
        return vis_data
    
    
    def plot_topic_distribution(self):
        """Plot topic distribution across categories"""
        if not self.topic_model_fitted or self.processed_data is None:
            print("Topic model not fitted or data not available!")
            return
        
        # Get topic probabilities for each document
        X_count = self.count_vectorizer.transform(self.processed_data['processed_text'])
        topic_probs = self.lda_topic_model.transform(X_count)
        
        # Create topic labels
        topic_labels = [f"Topic {i+1}" for i in range(self.lda_topic_model.n_components)]
        
        # Add topic probabilities to dataframe
        topic_df = pd.DataFrame(topic_probs, columns=topic_labels)
        topic_df['category'] = self.processed_data['category'].values
        
        # Create visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Topic Distribution by Category',
                'Average Topic Probability by Category',
                'Topic Heatmap',
                'Top Words per Topic'
            ),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "heatmap"}, {"type": "table"}]]
        )
        
        # Plot 1: Topic distribution by category
        for topic in topic_labels:
            category_topic_mean = topic_df.groupby('category')[topic].mean()
            fig.add_trace(
                go.Bar(x=category_topic_mean.index, y=category_topic_mean.values, 
                      name=topic, showlegend=False),
                row=1, col=1
            )
        
        # Plot 2: Average topic probability
        avg_topic_prob = topic_df[topic_labels].mean()
        fig.add_trace(
            go.Bar(x=topic_labels, y=avg_topic_prob.values, 
                  marker_color='lightblue', showlegend=False),
            row=1, col=2
        )
        
        # Plot 3: Heatmap of topics by category
        heatmap_data = topic_df.groupby('category')[topic_labels].mean()
        fig.add_trace(
            go.Heatmap(
                z=heatmap_data.values,
                x=topic_labels,
                y=heatmap_data.index,
                colorscale='Viridis',
                showscale=False
            ),
            row=2, col=1
        )
        
        # Plot 4: Top words per topic
        feature_names = self.count_vectorizer.get_feature_names_out()
        top_words_data = []
        
        for topic_idx in range(self.lda_topic_model.n_components):
            top_words_idx = self.lda_topic_model.components_[topic_idx].argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            top_words_data.append([f"Topic {topic_idx+1}", ", ".join(top_words[:5])])
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Topic', 'Top Words']),
                cells=dict(values=list(zip(*top_words_data)))
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="Topic Analysis Dashboard",
            showlegend=True
        )
        
        fig.show()
        return fig
    
    def create_sentiment_wordclouds(self):
        """Create word clouds for each sentiment category"""
        if self.processed_data is None:
            print("Data not available!")
            return
        
        categories = self.processed_data['category'].unique()
        n_categories = len(categories)
        
        fig, axes = plt.subplots(1, n_categories, figsize=(5*n_categories, 5))
        if n_categories == 1:
            axes = [axes]
        
        for i, category in enumerate(categories):
            category_text = ' '.join(
                self.processed_data[self.processed_data['category'] == category]['processed_text']
            )
            
            if len(category_text.strip()) > 0:
                wordcloud = WordCloud(
                    width=400, height=300,
                    background_color='white',
                    colormap='viridis',
                    max_words=50
                ).generate(category_text)
                
                axes[i].imshow(wordcloud, interpolation='bilinear')
                axes[i].set_title(f'{category} Sentiment', fontsize=14, fontweight='bold')
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('sentiment_wordclouds.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self, X_test, y_test, X_test_dense):
        """Plot model performance comparison"""
        # Get predictions
        knn_pred = self.knn_model.predict(X_test)
        lda_pred = self.lda_classifier.predict(X_test_dense)
        
        # Calculate accuracies
        knn_accuracy = accuracy_score(y_test, knn_pred)
        lda_accuracy = accuracy_score(y_test, lda_pred)
        
        # Create comparison plot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Accuracy Comparison', 'KNN Confusion Matrix', 
                          'LDA Confusion Matrix', 'Classification Report'),
            specs=[[{"type": "bar"}, {"type": "heatmap"}],
                   [{"type": "heatmap"}, {"type": "table"}]]
        )
        
        # Accuracy comparison
        fig.add_trace(
            go.Bar(x=['KNN', 'LDA'], y=[knn_accuracy, lda_accuracy],
                  marker_color=['lightcoral', 'lightblue']),
            row=1, col=1
        )
        
        # Confusion matrices
        knn_cm = confusion_matrix(y_test, knn_pred)
        lda_cm = confusion_matrix(y_test, lda_pred)
        
        categories = self.label_encoder.classes_
        
        fig.add_trace(
            go.Heatmap(z=knn_cm, x=categories, y=categories, 
                      colorscale='Blues', showscale=False),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Heatmap(z=lda_cm, x=categories, y=categories, 
                      colorscale='Reds', showscale=False),
            row=2, col=1
        )
        
        # Classification report table
        knn_report = classification_report(y_test, knn_pred, target_names=categories, output_dict=True)
        report_data = []
        for category in categories:
            report_data.append([
                category, 
                f"{knn_report[category]['precision']:.3f}",
                f"{knn_report[category]['recall']:.3f}",
                f"{knn_report[category]['f1-score']:.3f}"
            ])
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Category', 'Precision', 'Recall', 'F1-Score']),
                cells=dict(values=list(zip(*report_data)))
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Model Performance Analysis")
        fig.show()
        
        return fig
    
    def save_models(self, model_dir='saved_models'):
        """Save trained models"""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        models_to_save = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'count_vectorizer': self.count_vectorizer,
            'knn_model': self.knn_model,
            'lda_classifier': self.lda_classifier,
            'lda_topic_model': self.lda_topic_model,
            'label_encoder': self.label_encoder
        }
        
        for name, model in models_to_save.items():
            with open(f'{model_dir}/{name}.pkl', 'wb') as f:
                pickle.dump(model, f)
        
        print(f"Models saved to {model_dir}/")
    
    def load_models(self, model_dir='saved_models'):
        """Load saved models"""
        models_to_load = [
            'tfidf_vectorizer', 'count_vectorizer', 'knn_model',
            'lda_classifier', 'lda_topic_model', 'label_encoder'
        ]
        
        for name in models_to_load:
            try:
                with open(f'{model_dir}/{name}.pkl', 'rb') as f:
                    setattr(self, name, pickle.load(f))
                print(f"Loaded {name}")
            except FileNotFoundError:
                print(f"Model {name} not found in {model_dir}/")
        
        self.topic_model_fitted = True
    
    def predict_with_topics(self, text):
        """Predict sentiment and topic for new text"""
        processed_text = self.preprocess_text(text)
        
        # Sentiment prediction
        text_tfidf = self.tfidf_vectorizer.transform([processed_text])
        knn_pred = self.knn_model.predict(text_tfidf)
        knn_prob = self.knn_model.predict_proba(text_tfidf)
        
        # Topic prediction
        text_count = self.count_vectorizer.transform([processed_text])
        topic_probs = self.lda_topic_model.transform(text_count)
        dominant_topic = np.argmax(topic_probs[0])
        
        # Get top words for dominant topic
        feature_names = self.count_vectorizer.get_feature_names_out()
        top_words_idx = self.lda_topic_model.components_[dominant_topic].argsort()[-5:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        
        sentiment = self.label_encoder.inverse_transform(knn_pred)[0]
        
        return {
            'sentiment': sentiment,
            'sentiment_confidence': max(knn_prob[0]),
            'dominant_topic': f"Topic {dominant_topic + 1}",
            'topic_probability': topic_probs[0][dominant_topic],
            'topic_keywords': top_words
        }
    def clean_text(self, text):
        """Fungsi untuk normalisasi teks"""
        # cleaning text
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())

        # remove emojis
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.translate(str.maketrans('', '', string.digits))

        # tokenization
        tokens = word_tokenize(text)
        # stemming & stopword removal
        tokens = [self.stemmer.stem(token) for token in tokens
                 if token not in self.stop_words and len(token) > 2]
        # remove mentions
        text = ' '.join(tokens)
        return text

    def classify(self, text):
        """Fungsi untuk mengklasifikasikan teks"""

        # Try to load existing models
        try:
            self.load_models()
            print("Loaded existing models!")
        except:
            print("Training new models...")
            df = self.load_data()
            if df is None:
                return
            X_tfidf, X_count, y, _ = self.prepare_data(df)
            self.train_classification_models(X_tfidf, y)
            self.train_topic_model(X_count)
            self.save_models()
        
        print("\n=== INTERACTIVE SENTIMENT & TOPIC ANALYSIS ===")
        result = self.predict_with_topics(text)
        print(f"\n📊 Analysis Results:")
        print(f"   Sentiment: {result['sentiment']} (confidence: {result['sentiment_confidence']:.4f})")
        print(f"   Dominant Topic: {result['dominant_topic']} (probability: {result['topic_probability']:.4f})")
        print(f"   Topic Keywords: {', '.join(result['topic_keywords'])}")
        return result['sentiment']
def main():
    """Main execution function"""
    print("=== SENTIMENT ANALYSIS WITH VISUALIZATION ===")
    
    # Initialize analyzer
    analyzer = SentimentClassifier()
    
    # Load data
    print("Loading data from database...")
    df = analyzer.load_data()
    if df is None:
        print("Failed to load data!")
        return
    
    # Prepare data
    X_tfidf, X_count, y, processed_df = analyzer.prepare_data(df)
    
    # Train classification models
    X_train, X_test, y_train, y_test, X_train_dense, X_test_dense = analyzer.train_classification_models(X_tfidf, y)
    
    # Train topic model
    analyzer.train_topic_model(X_count)
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # 1. LDAvis interactive visualization
    analyzer.create_ldavis_visualization(X_count)
    
    # 2. Topic distribution plots
    analyzer.plot_topic_distribution()
    
    # 3. Sentiment word clouds
    analyzer.create_sentiment_wordclouds()
    
    # 4. Model comparison plots
    analyzer.plot_model_comparison(X_test, y_test, X_test_dense)
    
    # Save models
    analyzer.save_models()
    
    # Example predictions
    print("\n=== EXAMPLE PREDICTIONS WITH TOPICS ===")
    test_texts = [
        "Produk ini sangat bagus dan memuaskan sekali",
        "Pelayanan buruk sekali, sangat mengecewakan",
        "Produk standar, tidak istimewa tapi juga tidak buruk"
    ]
    
    for text in test_texts:
        result = analyzer.predict_with_topics(text)
        print(f"\nText: '{text}'")
        print(f"Sentiment: {result['sentiment']} (confidence: {result['sentiment_confidence']:.4f})")
        print(f"Dominant Topic: {result['dominant_topic']} (probability: {result['topic_probability']:.4f})")
        print(f"Topic Keywords: {', '.join(result['topic_keywords'])}")

if __name__ == "__main__":
    main()

# Fungsi tambahan untuk prediksi interaktif
def interactive_prediction():
    """Interactive prediction with topic analysis"""
    analyzer = SentimentClassifier()
    
    # Try to load existing models
    try:
        analyzer.load_models()
        print("Loaded existing models!")
    except:
        print("Training new models...")
        df = analyzer.load_data()
        if df is None:
            return
        X_tfidf, X_count, y, _ = analyzer.prepare_data(df)
        analyzer.train_classification_models(X_tfidf, y)
        analyzer.train_topic_model(X_count)
        analyzer.save_models()
    
    print("\n=== INTERACTIVE SENTIMENT & TOPIC ANALYSIS ===")
    print("Type 'quit' to exit")
    
    while True:
        text = input("\nEnter text for analysis: ")
        if text.lower() == 'quit':
            break
        
        result = analyzer.predict_with_topics(text)
        print(f"\n📊 Analysis Results:")
        print(f"   Sentiment: {result['sentiment']} (confidence: {result['sentiment_confidence']:.4f})")
        print(f"   Dominant Topic: {result['dominant_topic']} (probability: {result['topic_probability']:.4f})")
        print(f"   Topic Keywords: {', '.join(result['topic_keywords'])}")
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