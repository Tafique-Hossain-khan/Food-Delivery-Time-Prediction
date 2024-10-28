FROM quay.io/astronomer/astro-runtime:12.2.0

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["streamlit", "run", "app.py"]
