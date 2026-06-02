FROM python:3.11-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock* README.md ./
COPY src ./src

RUN uv sync --all-groups

CMD ["uv", "run", "python", "-m", "src.ingestion.opensky_client"]