FROM python:3.8.5-alpine

COPY requirements.txt /
RUN pip install -r requirements.txt

WORKDIR /srv

COPY ./server /srv

ENV FLASK_ENV=production

CMD ["uwsgi", "--ini", "wsgi.ini"]