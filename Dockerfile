FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/home/app/.cache/huggingface

RUN useradd --create-home app
WORKDIR /srv

COPY requirements-core.txt requirements.txt ./
# CPU-only torch first: the default wheel drags in several GB of CUDA libraries.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
USER app
EXPOSE 8000

# GROQ_API_KEY must be provided at runtime: docker run --env-file .env ...
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
