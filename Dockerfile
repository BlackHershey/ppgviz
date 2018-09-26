FROM python:3.6-slim

WORKDIR /app

COPY ./requirements.txt /app/

RUN pip install -r /app/requirements.txt

COPY . /app


CMD ["python", "pet_aif_viz.py"]


