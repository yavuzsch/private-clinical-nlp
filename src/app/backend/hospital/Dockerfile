FROM clinical-nlp-base:latest

WORKDIR /app

COPY src/app/backend/shared ./src/app/backend/shared
COPY src/app/backend/hospital ./src/app/backend/hospital

ENV PYTHONPATH=/app/src/app/backend

EXPOSE 8100

CMD ["sh", "-c", "uvicorn src.app.backend.hospital.main:app --host 0.0.0.0 --port $((8100 + HOSPITAL_ID))"]