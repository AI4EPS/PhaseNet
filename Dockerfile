FROM tensorflow/tensorflow:2.3.1

RUN pip install tqdm obspy pandas minio

WORKDIR /opt
COPY ./phasenet/*py /opt/
COPY ./model /opt/model

#ENTRYPOINT ["python"]
