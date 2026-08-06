FROM python:3.10-slim-bullseye
RUN apt-get update && apt-get install -y git ssh && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . /app
RUN pip install .
EXPOSE 5903/tcp
CMD ["distopf-federate-server"]

