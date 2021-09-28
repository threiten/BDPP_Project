FROM python:slim
RUN apt-get update && apt-get -y upgrade \
  && apt-get install -y --no-install-recommends \
  git \
  wget \
  g++ \
  ca-certificates \
  && rm -rf /var/lib/apt/lists/*

ENV PATH "/usr/miniconda3/bin:${PATH}"
ARG PATH="/usr/miniconda3/bin:${PATH}"

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /usr/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /usr/miniconda3 \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && echo "Running $(conda --version)" && \
    conda update conda && \
    conda install python=3.8 pip

RUN mkdir -p /deploy
COPY requirements.txt deploy/requirements.txt
RUN conda install -c conda-forge -c plotly -c pytorch -y --file /deploy/requirements.txt

COPY app.py /deploy/app.py
COPY LSTMmodel.py /deploy/LSTMmodel.py
COPY utils.py /deploy/utils.py
COPY gunicorn_config.py /deploy/gunicorn_config.py

RUN mkdir -p /deploy/mplcache
ENV MPLCONFIGDIR "/deploy/mplcache"

WORKDIR /deploy

EXPOSE 8080

CMD ["gunicorn", "--config", "/deploy/gunicorn_config.py", "app:server"]
