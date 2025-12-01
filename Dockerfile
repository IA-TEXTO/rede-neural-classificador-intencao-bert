FROM python:3.11

RUN useradd --create-home myuser

RUN python3 -m venv /home/myuser/venv
ENV VIRTUAL_ENV=/home/myuser/venv
ENV PATH="/home/myuser/venv/bin:$PATH"

WORKDIR /home/myuser/code
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip3 install --no-cache-dir wheel
RUN pip3 install --no-cache-dir -r requirements.txt

USER myuser

COPY . .

EXPOSE 5000
CMD ["python", "main.py"]
