FROM clinical-nlp-base:latest

WORKDIR /app

COPY src/app/backend/shared ./src/app/backend/shared
COPY src/app/backend/central ./src/app/backend/central

ENV PYTHONPATH=/app/src/app/backend

EXPOSE 8000

CMD ["uvicorn", "src.app.backend.central.main:app", "--host", "0.0.0.0", "--port", "8000"]