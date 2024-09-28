FROM python:3.12
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
VOLUME /app/data
CMD ["python", "run_demo.py"]
