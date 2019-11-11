FROM continuumio/miniconda3

ENTRYPOINT [ "/bin/bash", "-c" ]

EXPOSE 5000

RUN set -x \
     && apt-get update \
     && apt-get install --no-install-recommends --no-install-suggests -y libpq-dev build-essential \
     && rm -rf /var/lib/apt/lists/*

# update conda, pre-install BentoML base dependencies
RUN conda update conda -y \
      && conda install pip numpy scipy \
      && pip install gunicorn six

# copy over model files
COPY . /bento
WORKDIR /bento

# update conda base env
RUN conda env update -n base -f /bento/environment.yml
RUN pip install -r /bento/requirements.txt

 # Install additional pip dependencies inside bundled_pip_dependencies dir
RUN if [ -f /bento/bentoml_init.sh ]; then /bin/bash -c /bento/bentoml_init.sh; fi

# run user defined setup script
RUN if [ -f /bento/setup.sh ]; then /bin/bash -c /bento/setup.sh; fi

# Run Gunicorn server with path to module.
CMD ["bentoml serve-gunicorn /bento"]
