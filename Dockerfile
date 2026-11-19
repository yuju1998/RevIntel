# RevIntel API image (FastAPI + CRM, Redis cache, MLflow champion scoring,
# sentence-transformers semantic search, OpenAI Copilot).
#
#   docker compose up --build        # see docs/DOCKER.md
#
# Nothing secret or local is baked in: settings and keys come from the environment at
# runtime, MLflow models are loaded lazily from the tracking server. The one model in the
# image is the pinned embedding model (EMBEDDING_MODEL build arg), because the container
# runs with HF_HUB_OFFLINE=1: the runtime EMBEDDING_MODEL must equal the baked one, and
# changing the model requires rebuilding the image.

ARG PYTHON_IMAGE=python:3.14.8-slim-trixie
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# --- dependencies (cached until a requirements file changes) ---------------------------------
FROM ${PYTHON_IMAGE} AS deps

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH=/opt/venv/bin:$PATH

RUN python -m venv /opt/venv
COPY requirements/base.txt requirements/api.txt requirements/lock.txt /tmp/requirements/

# CPU-only builds at the locked versions: torch from the PyTorch CPU index (the PyPI Linux
# wheel pulls CUDA) and xgboost-cpu (the `xgboost` Linux wheel pulls NCCL). Their
# dependencies then come from PyPI with every other package, pinned by the lock.
RUN lock=/tmp/requirements/lock.txt \
    && torch_version="$(grep -E '^torch==' "$lock" | cut -d= -f3)" \
    && xgboost_version="$(grep -E '^xgboost==' "$lock" | cut -d= -f3)" \
    && pip install --no-deps --index-url https://download.pytorch.org/whl/cpu "torch==${torch_version}" \
    && pip install --no-deps "xgboost-cpu==${xgboost_version}" \
    && pip install -r /tmp/requirements/api.txt -c "$lock" \
    && pip check

# Bake the embedding model into the Hugging Face cache (no download at runtime).
ARG EMBEDDING_MODEL
ENV HF_HOME=/opt/huggingface
RUN python -c "import sys; from sentence_transformers import SentenceTransformer; \
model = SentenceTransformer(sys.argv[1], device='cpu'); \
assert model.get_embedding_dimension() == 384, model.get_embedding_dimension()" \
    "${EMBEDDING_MODEL}"

# --- runtime ---------------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime

# libgomp1: OpenMP runtime used by XGBoost.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --create-home --home-dir /home/app app

ARG EMBEDDING_MODEL
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app:/app/backend \
    HF_HOME=/opt/huggingface \
    HF_HUB_OFFLINE=1 \
    BAKED_EMBEDDING_MODEL=${EMBEDDING_MODEL} \
    EMBEDDING_MODEL=${EMBEDDING_MODEL}

COPY --from=deps /opt/venv /opt/venv
# Owned by the app user: Hugging Face may write lock/marker files even when offline.
COPY --from=deps --chown=app:app /opt/huggingface /opt/huggingface

WORKDIR /app
COPY alembic.ini ./
COPY backend/alembic backend/alembic
COPY ml ml
COPY backend/app backend/app

USER app
EXPOSE 8000

CMD ["fastapi", "run", "backend/app/main.py", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
