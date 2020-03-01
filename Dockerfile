FROM python:3.6

COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN pip install --no-cache-dir uwsgi
CMD [ "python" ]

