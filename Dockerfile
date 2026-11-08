FROM python:3.13-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app app
COPY scripts scripts
COPY docker-entrypoint.sh .
ENV DATA_DIR=/srv/data
EXPOSE 8000
ENTRYPOINT ["./docker-entrypoint.sh"]
