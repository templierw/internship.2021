## / dis_report_version

# Disaggregation: report version

This directory contains the notebooks used to produce the results presented in the report and during my defense in June 2021.

# Try the code

A public Docker containter is available to try the code:

```bash
docker pull pasokon2dev/internship.2021:disaggregation1.0
docker run -p 8888:8888 pasokon2dev/internship.2021:disaggregation1.0 
```

You can also clone the repository, but then need to set up a few environment variables using your CLI:

```bash
$<project/root/directory/>~> export PYTHONPATH=$PWD/lib
$<project/root/directory/>~> export DATA_DIR=$PWD/data/phase01
$<project/root/directory/>~> export VIZ_DIR=$PWD/work/dis_report_version/plots
$<project/root/directory/>~> export RES_DIR=$PWD/work/dis_report_version/results
```

and install the python librairies:
```bash
pip3 install -r requirements.txt
```