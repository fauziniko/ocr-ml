# Gunakan image Python versi terbaru
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app

# Buat direktori kerja
WORKDIR $APP_HOME

# Salin file aplikasi ke container
COPY . $APP_HOME/

# Instal dependensi
RUN pip install --no-cache-dir -r requirements.txt

# Expose port Flask
EXPOSE 5000

# Jalankan aplikasi Flask
CMD ["python", "main.py"]
