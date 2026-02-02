# Używamy lekkiej wersji Pythona, żeby oszczędzać RAM
FROM python:3.9-slim

# Ustawiamy folder roboczy w kontenerze
WORKDIR /app

# Najpierw kopiujemy tylko zależności (dzięki temu Docker je "cache'uje")
COPY requirements.txt .

# Instalujemy biblioteki
RUN pip install --no-cache-dir -r requirements.txt

# Kopiujemy resztę plików aplikacji
COPY . .

# Informujemy, że aplikacja używa portu 5000 (standard dla Flask)
EXPOSE 5000

# Komenda startowa
CMD ["python", "app.py"]