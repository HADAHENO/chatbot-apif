# استخدمي Python كبيئة
FROM python:3.10

# تعيين مجلد العمل
WORKDIR /app

# نسخ الملفات
COPY . /app

# تثبيت المتطلبات
RUN pip install --no-cache-dir -r requirements.txt

# تشغيل التطبيق
CMD ["python", "main.py"]
