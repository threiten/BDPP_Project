FROM python:slim
RUN apt-get update && apt-get -y upgrade \
  && apt-get install -y --no-install-recommends \
  git \
  wget \
  g++ \
  ca-certificates \
  && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && echo "Running $(conda --version)" && \
    conda init bash && \
    . /root/.bashrc && \
    conda update conda && \
    conda create -n partypredictor && \
    conda activate partypredictor && \
    conda install python=3.8 pip

RUN mkdir -p /deploy
COPY requirements.txt deploy/requirements.txt
RUN conda install -c conda-forge -c plotly -c pytorch -y --file /deploy/requirements.txt

COPY app.py /deploy/app.py
COPY LSTMmodel.py /deploy/LSTMmodel.py
COPY utils.py /deploy/utils.py
COPY gunicorn_config.py /deploy/gunicorn_config.py

WORKDIR /deploy

EXPOSE 8080

RUN echo 'conda activate partypredictor' >> /root/.bashrc

ENTRYPOINT ["bin/bash", "-l", "-c"]

CMD ["gunicorn", "--config", "/deploy/gunicorn_config.py", "app:server"]
