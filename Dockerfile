FROM python:3.10-slim


WORKDIR /app

COPY flask_app/ /app/

COPY models/count_vectorizer.pkl /app/models/count_vectorizer.pkl


RUN pip install -r requirements.txt

RUN python -m nltk.downloader stopwords wordnet

EXPOSE 5000

CMD ["python","app.py"] 