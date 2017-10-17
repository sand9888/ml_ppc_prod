FROM ubuntu:latest 


MAINTAINER Narnik Gamarnik <narnikgamarnikus@gmail.com>

ADD ayur_all_users.csv \
	ga_ayur.py \
	app.py \
	xgboost_cpc.py \ 
	/forsolving.com /forsolving.com/ 

RUN apt-get update \
	&& apt-get install -y python3-pip python3-dev \
 	libcurl4-gnutls-dev libexpat1-dev gettext \
  	libz-dev libssl-dev git curl wget python3-tk \
  	&& cd /usr/local/bin \
  	&& ln -s /usr/bin/python3 python \
  	&& pip3 install --upgrade pip

RUN pip3 install --no-cache-dir \
	'numpy==1.13.0' \ 
	'pandas==0.19.2' \
	'matplotlib==2.0.2' \
	'seaborn==0.7.1' \
	'graphviz==0.5.2' \
	'scikit-learn==0.18.2' \
	'sanic' 

RUN git clone --recursive https://github.com/dmlc/xgboost && \
    cd xgboost && \
    make -j4 && \
	cd python-package; python3 setup.py install

RUN pip3 install --no-cache-dir \
	'xgboost==0.6a2' 

CMD [ "python3", "./app.py" ]