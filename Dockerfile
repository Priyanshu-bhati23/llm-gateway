FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY gateway ./gateway
RUN pip install --no-cache-dir . && useradd --create-home runner
USER runner
ENV GATEWAY_CONFIG=/config/gateway.yaml
EXPOSE 8080
CMD ["uvicorn", "gateway.app:app_from_env", "--factory", "--host", "0.0.0.0", "--port", "8080"]
