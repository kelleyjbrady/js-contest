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

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install Poetry using the specified version
RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /app

COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-root

COPY ./src ./src

CMD ["tail", "-f", "/dev/null"]