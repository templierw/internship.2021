FROM python:3.8

WORKDIR /extension

COPY ./work/post_defense_work /extension/notebooks
COPY ./data/phase02 /extension/data
COPY ./lib/modelUtils.py /extension/lib/modelUtils.py
COPY ./lib/distorch /extension/lib/distorch

RUN pip install pip pipenv --upgrade
RUN pipenv install jupyter numpy matplotlib pandas scikit-learn torch tensorboard

ENV PYTHONPATH="/extension/lib"

ENV ROOT_DIR="/extension"
ENV DATA_DIR="/extension/data"

ENV MOD_DIR="/extension/notebooks/saved_models"
ENV RES_DIR="/extension/notebooks/results"

CMD ["pipenv", "run", "jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]