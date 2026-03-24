# We use ARG before FROM to specify the base image tag dynamically
ARG PYTHON_VERSION=3.13
FROM python:${PYTHON_VERSION}-slim

# Accept the Poetry version as an argument
ARG POETRY_VERSION=1.8.2

# Set environment variables for Poetry
ENV POETRY_VERSION=${POETRY_VERSION} \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

# Prepend Poetry to path
ENV PATH="$POETRY_HOME/bin:$PATH"

# Install system dependencies, including those required for the gcloud repository
RUN apt-get update && apt-get install -y \
    curl \
    apt-transport-https \
    ca-certificates \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Add Google Cloud public key, repository, and install the gcloud CLI
RUN curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && apt-get update \
    && apt-get install -y google-cloud-cli \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry using the specified version
RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /app

COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-root

COPY ./src ./src

CMD ["tail", "-f", "/dev/null"]