FROM python:3.9-slim
RUN apt-get update && apt-get -y upgrade \
  && apt-get install -y --no-install-recommends \
  git \
  git-lfs \
  wget \
  g++ \
  ca-certificates \
  && rm -rf /var/lib/apt/lists/*

ENV PATH "/usr/miniconda3/bin:${PATH}"
ARG PATH="/usr/miniconda3/bin:${PATH}"

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_22.11.1-1-Linux-x86_64.sh \
    && mkdir /usr/.conda \
    && bash Miniconda3-py39_22.11.1-1-Linux-x86_64.sh -b -p /usr/miniconda3 \
    && rm -f Miniconda3-py39_22.11.1-1-Linux-x86_64.sh \
    && echo "Running $(conda --version)" && \
    conda update conda && \
    conda install python=3.9 pip

RUN mkdir -p /deploy
COPY requirements.txt deploy/requirements.txt
RUN conda install -k -S -c conda-forge -c plotly -c pytorch -c huggingface -y --file /deploy/requirements.txt

COPY app.py /deploy/app.py
COPY LSTMmodel.py /deploy/LSTMmodel.py
COPY utils.py /deploy/utils.py
COPY gunicorn_config.py /deploy/gunicorn_config.py

RUN mkdir -p /deploy/mplcache
ENV MPLCONFIGDIR "/deploy/mplcache"

WORKDIR /deploy

RUN git lfs install
RUN git clone https://huggingface.co/datasets/threite/Bundestag-v2
RUN git clone https://huggingface.co/threite/xlm-roberta-base-finetuned-partypredictor-test

EXPOSE 8080

CMD ["gunicorn", "--timeout", "1000", "--config", "/deploy/gunicorn_config.py", "app:server"]
