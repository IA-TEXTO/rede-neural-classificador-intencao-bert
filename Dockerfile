FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN useradd -m myuser

WORKDIR /home/myuser/app

COPY requirements.txt .

RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

USER myuser

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

