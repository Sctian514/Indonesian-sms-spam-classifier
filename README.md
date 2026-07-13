📱 Indonesian SMS Spam Detection using TF-IDF & Random Forest

A web-based machine learning application for detecting spam messages in Indonesian using Natural Language Processing (NLP) and the Random Forest classification algorithm.

This project was developed as part of my undergraduate thesis in Software Engineering and demonstrates the implementation of text preprocessing, feature extraction, and machine learning for SMS spam classification.

🚀 Live Demo

Coming Soon

📷 Application Preview

Add screenshots of your application here.

<img width="626" height="338" alt="image" src="https://github.com/user-attachments/assets/1469b6aa-b9fa-4632-9f0a-b9ce7ae5c000" />

<img width="432" height="878" alt="image" src="https://github.com/user-attachments/assets/101c954b-f97c-4b3a-9281-93e10e36fb54" />

<img width="555" height="865" alt="image" src="https://github.com/user-attachments/assets/7704ae34-7666-4fe4-ad6b-044021a9007c" />

Example:
Home Page
Prediction Result
Spam Type Detection
✨ Features
Detect whether an SMS message is Spam or Non-Spam
Classify spam into several categories
Promotion
Fraud
Online Gambling
Loan (KTA)
Indonesian text preprocessing
Machine Learning prediction using Random Forest
Fast and easy-to-use web interface
🧠 Machine Learning Pipeline

The application follows the following workflow:

Input SMS message
Text preprocessing
Case Folding
Cleaning
Tokenization
Stopword Removal
Stemming (Sastrawi)
TF-IDF Feature Extraction
Random Forest Classification
Display prediction result
🛠 Tech Stack
Backend
Python
Flask
Scikit-learn
Pandas
NumPy
Natural Language Processing
NLTK
Sastrawi
TF-IDF
Machine Learning
Random Forest
Frontend
HTML
CSS
JavaScript
📂 Project Structure
spam-detection/
│
├── app.py
├── requirements.txt
├── model/
│   ├── spam_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
└── README.md

📊 Dataset

The dataset consists of Indonesian SMS messages collected from publicly available sources and manually labeled into three classes:

Label	Description
Non-Spam	Normal messages
Spam Promotion	Promotional messages
Harmful Spam	Fraud, Gambling, Loan, etc.

The dataset was preprocessed before training to improve classification performance.


🎯 Future Improvements
Deploy using Render
REST API using Flask/FastAPI
Mobile-friendly UI
Deep Learning model (IndoBERT)
Explainable AI for prediction
User authentication
Prediction history

👨‍💻 Author

Samuel Cristian Saragih

Software Engineer | Machine Learning Enthusiast | NLP Enthusiast

📄 License

This project is licensed under the MIT License.
