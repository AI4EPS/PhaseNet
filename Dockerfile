FROM tensorflow/tensorflow:2.14.0

# Create the environment:
# COPY env.yml /app
# RUN conda env create --name cs329s --file=env.yml
# Make RUN commands use the new environment:
# SHELL ["conda", "run", "-n", "cs329s", "/bin/bash", "-c"]

RUN pip install tqdm obspy pandas 
RUN pip install uvicorn fastapi
RUN pip install fsspec gcsfs s3fs

WORKDIR /opt

# Copy files
COPY phasenet /opt/phasenet
COPY model /opt/model
COPY application_default_credentials.json /opt/application_default_credentials.json

# Expose API port
EXPOSE 8000

ENV PYTHONUNBUFFERED=1

# Start API server
#ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "cs329s", "uvicorn", "--app-dir", "phasenet", "app:app", "--reload", "--port", "8000", "--host", "0.0.0.0"]
# ENTRYPOINT ["uvicorn", "--app-dir", "phasenet", "app:app", "--reload", "--port", "8000", "--host", "0.0.0.0"]
