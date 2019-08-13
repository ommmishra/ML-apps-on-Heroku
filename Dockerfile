FROM python:3.7-slim-stretch

RUN apt update
RUN apt install -y python3-dev gcc

RUN pip3 install https://download.pytorch.org/whl/cpu/torch-1.1.0-cp37-cp37m-linux_x86_64.whl
RUN pip3 install fastai==1.0.52
RUN pip3 install numpy==1.16.3

RUN pip3 install flask

COPY app /app	

CMD ['python3', '/app/hello.py']