FROM python:3.9-slim
#install lib
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

# Port
EXPOSE 5000

#Run Flask
CMD ["python", "app.py"]
