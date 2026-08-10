FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock
COPY . .
EXPOSE 8000
CMD ["uvicorn", "doc_agent.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
