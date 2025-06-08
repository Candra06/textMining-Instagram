
from flask import Flask, render_template, flash, request, url_for, jsonify, request, send_file, redirect, session
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from dotenv import load_dotenv
import os
from Module.Instat import Instat
import matplotlib.pyplot as plt
from Module.SentimentClassifier import SentimentClassifier
from wordcloud import WordCloud
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, Column, Integer, String, DateTime
import io
import base64
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import cryptography
import scipy
# from gensim import corpora
# from gensim.models import LdaModel
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# try:
#     nltk.data.find('corpora/stopwords')
# except LookupError:
nltk.download('stopwords')




app = Flask(__name__)
# env
DATABASE_NAME = os.getenv('DATABASE_NAME')  or 'sentiment_analysis'
DATABASE_USER = os.getenv('DATABASE_USER') or 'root'
DATABASE_PASSWORD = os.getenv('DATABASE_PASSWORD') or 'root'
DATABASE = 'mysql+pymysql://'+DATABASE_USER+':'+DATABASE_PASSWORD+'@localhost/'+DATABASE_NAME
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE
engine = create_engine(DATABASE, echo=True)
db = SQLAlchemy(app)
migrate = Migrate(app, db)
load_dotenv()
Session = sessionmaker(bind=engine)
session = Session()
# models
class Training(db.Model):
    # Primary key for the User table
    id = db.Column(db.Integer, primary_key=True)
    # description without maximum characters
    description = db.Column(db.Text, nullable=True)
    # category using enumeration(Positif, Negatif, Netral)
    category = db.Column(db.Enum('Positif', 'Negatif', 'Netral'), nullable=False, default='Netral')

class Post(db.Model):
    # Primary key for the Post table
    id = db.Column(db.Integer, primary_key=True)
    # title of the post with maximum 100 characters
    title = db.Column(db.String(100), nullable=False)
    # link post
    link = db.Column(db.String(200), nullable=False)
    # category
    category = db.Column(db.Enum('Positif', 'Negatif', 'Netral'), nullable=False, default='Netral')
    # file comments
    comments = db.Column(db.Text, nullable=True)

class Testing(db.Model):
    # Primary key for the User table
    id = db.Column(db.Integer, primary_key=True)
    # description without maximum characters
    description = db.Column(db.Text, nullable=True)
    # category using enumeration(Positif, Negatif, Netral)
    category = db.Column(db.Enum('Positif', 'Negatif', 'Netral'), nullable=False, default='Netral')
    # make relationship with Post table
    post_id = db.Column(db.Integer, db.ForeignKey('post.id'), nullable=False)
    # relationship with Post table
    post = db.relationship('Post', backref=db.backref('testings', lazy=True))

    username = db.Column(db.String(80), nullable=True)


    # String representation of the User object
    def __repr__(self):
        return f'<User {self.name}>'


@app.route('/')
def index():
    return render_template('frontend/index.html')
def clean_text(text):
    """Clean and preprocess text for word cloud generation"""
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    return ' '.join(filtered_tokens)
def generate_wordcloud(text, width=800, height=400, max_words=100, background_color='white'):
    """Generate word cloud from text"""
    if not text.strip():
        return None
    
    # Clean the text
    cleaned_text = clean_text(text)
    
    if not cleaned_text.strip():
        return None
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=width,
        height=height,
        max_words=max_words,
        background_color=background_color,
        colormap='viridis',
        relative_scaling=0.5,
        random_state=42
    ).generate(cleaned_text)

    

    buffer = io.BytesIO()
    wordcloud.to_image().save(buffer, 'png')
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    print("Image Base64:", img_base64)
    return img_base64

@app.route('/api/get_comments', methods=['POST'])
def get_comments():
    instat = Instat()
    url = request.json.get('url')
    post_id = instat.getPostId(url)
    file = post_id+"_Comments"+'.csv'
    # check if file exists
    # if  os.path.exists("Data/"+file):
        # delete file
    #     os.remove("Data/"+file)
    # instat.scrapByLink(url)
    # check by url
    post = session.query(Post).filter_by(link=url).first()
    if post:
        # loop csv file and insert data into testing table
        path = os.path.abspath("Data/"+file)
        print("File path:", path)
        commentsData = []
        texts= {
            "positif": [],
            "negatif": [],
            "netral": []
        }
        with open(path, 'r') as f:
            comments = f.readlines()[1:]
            for comment in comments:
                # skip empty comments
                if not comment.strip():
                    continue
                try:
                    # get comment by field text
                    text = comment.split(';')[3].replace('"', '').replace("'", "")
                    username = comment.split(';')[5].replace('"', '').replace("'", "")
                    text = SentimentClassifier().clean_text(text.strip())
                    testing = session.query(Testing).filter_by(description=text.strip(), post_id=post.id).first()
                    if testing:
                        if testing.category == 'Positif':
                            texts["positif"].append(text.strip())
                        elif testing.category == 'Negatif':
                            texts["negatif"].append(text.strip())
                        else:
                            texts["netral"].append(text.strip())
                        childData = {
                            'username': username,
                            "text": text.strip(),
                            'timestamp': '',
                            "likes": 0,
                            "category": testing.category,
                            "post_id": post.id  
                        }
                        commentsData.append(childData)
                        continue
                    category = SentimentClassifier().classify(text.strip())
                    testing = Testing(
                        description=text.strip(),
                        category=category,
                        post_id=post.id,
                        username=username
                    )
                    if category == 'Positif':
                        texts["positif"].append(text.strip())
                    elif category == 'Negatif':
                        texts["negatif"].append(text.strip())
                    else:
                        texts["netral"].append(text.strip())
                    childData = {
                        'username': username,
                        "text": text.strip(),
                        'timestamp': '',
                        "likes": 0,
                        "category": category,
                        "post_id": post.id  
                    }
                    commentsData.append(childData)
                    session.add(testing)
                except Exception as e:
                    print("Error:", e)
                    continue
            session.commit()
        print("Comments Data:", commentsData)
        allComments = {
            "positif": " ".join(texts["positif"]),
            "negatif": " ".join(texts["negatif"]),
            "netral": " ".join(texts["netral"])
        }
        print("All Comments:", allComments)
        # print("WORDCLOUD:", generate_wordcloud(allComments))
        return jsonify({
            'status': 'success',
            'data': {
                'comments': commentsData,
                "allComments": allComments,
                'wordcloud':{
                    "positif": "data:image/png;base64," + generate_wordcloud(allComments["positif"]),
                    "negatif": "data:image/png;base64," + generate_wordcloud(allComments["negatif"]),
                    "netral": "data:image/png;base64," + generate_wordcloud(allComments["netral"])
                },
                'sentimens': {
                    'positif': session.query(Testing).filter_by(category='Positif', post_id=post.id).count(),
                    'negatif': session.query(Testing).filter_by(category='Negatif', post_id=post.id).count(),
                    'netral': session.query(Testing).filter_by(category='Netral', post_id=post.id).count()
                },
                # "lda": visualize_lda(post.id)
            }
        })
    
    # insert new data into Post table using table
    post = Post(
        title='Title of Post',
        link=url,
        category='netral',
        comments=file
    )
    session.add(post)
    session.commit()
    path = os.path.abspath("Data/"+file)
    print("File path:", path)
    commentsData = []
    texts= [];
    with open(path, 'r') as f:
        comments = f.readlines()[1:]
        for comment in comments:
            # skip empty comments
            if not comment.strip():
                continue
            try:
                # get comment by field text
                text = comment.split(';')[3].replace('"', '').replace("'", "")
                text = SentimentClassifier().clean_text(text.strip())
                username = comment.split(';')[5].replace('"', '').replace("'", "")
                testing = session.query(Testing).filter_by(description=text.strip(), post_id=post.id).first()
                if testing:
                    texts.append(text.strip())
                    childData = {
                        'username': username,
                        "text": text.strip(),
                        'timestamp': '',
                        "likes": 0,
                        "category": testing.category,
                        "post_id": post.id  
                    }
                    commentsData.append(childData)
                    continue
                category = SentimentClassifier().classify(text.strip())
                testing = Testing(
                    description=text.strip(),
                    category=category,
                    post_id=post.id,
                    username=username
                )
                texts.append(text.strip())
                childData = {
                    'username': username,
                    "text": text.strip(),
                    'timestamp': '',
                    "likes": 0,
                    "category": category,
                    "post_id": post.id  
                }
                commentsData.append(childData)
                session.add(testing)
            except Exception as e:
                print("Error:", e)
                continue
        session.commit()
        print("Comments Data:", commentsData)
        allComments = " ".join(str(x) for x in texts)
    print("IMG BASE64:", generate_wordcloud(allComments))
    return jsonify({
        'status': 'success get data',
        'data': {
            'comments': commentsData,
                "allComments": allComments,
                'wordcloud':"data:image/png;base64,"+ generate_wordcloud(allComments),
                'sentimens': {
                    'positif': session.query(Testing).filter_by(category='Positif', post_id=post.id).count(),
                    'negatif': session.query(Testing).filter_by(category='Negatif', post_id=post.id).count(),
                    'netral': session.query(Testing).filter_by(category='Netral', post_id=post.id).count()
                },
        }
    })
def visualize_lda(id):
    """Visualize LDA topics using pyLDAvis"""
    

    # Load the data
    texts = [word_tokenize(text) for text in session.query(Testing.description).filter_by(post_id=id).all()]
    
    # Create a dictionary and corpus
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    # Train the LDA model
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=3, passes=15, random_state=42)
    
    # Visualize the topics
    vis = gensimvis.prepare(lda_model, corpus, dictionary)
    
    # Render the visualization to HTML
    html = pyLDAvis.prepared_data_to_html(vis)
    
    return html
if __name__ == "__main__":
    app.run(debug = True)