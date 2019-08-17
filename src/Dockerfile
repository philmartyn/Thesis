FROM python:3.7.0-stretch
COPY . /Backend/src
WORKDIR /Backend/src
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
EXPOSE 5672
RUN export PYTHONPATH=/Backend/
CMD python3 ./predictor/message-receiver.py