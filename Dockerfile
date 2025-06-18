FROM python:3.8.13

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

ENV PORT=8080

CMD ["python", "app_v1.2.0.py"]
