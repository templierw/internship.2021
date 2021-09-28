FROM python:3.8-slim

WORKDIR /disaggregation

COPY ./work/dis_report_version /disaggregation/notebooks
COPY ./data/phase01 /disaggregation/data
COPY ./lib/modelUtils.py /disaggregation/lib/modelUtils.py
COPY ./lib/pbfcast /disaggregation/lib/pbfcast

RUN pip install pip pipenv --upgrade
RUN pipenv install jupyter numpy matplotlib pandas scikit-learn mxnet gluonts

ENV PYTHONPATH="/disaggregation/lib"

ENV ROOT_DIR="/disaggregation"
ENV DATA_DIR="/disaggregation/data"
ENV VIZ_DIR="/disaggregation/notebooks/plots"
ENV RES_DIR="/disaggregation/notebooks/results"

CMD ["pipenv", "run", "jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]