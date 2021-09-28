# post_defense_work/

The work contained herein represents what I did from mid-June 2021 (after the report and defense) until mid-July - when I stopped to help a PhD with some data processing task. I shall briefly present the diverse notebooks and the library going with this folder (`lib/distorch`).

## Disaggregation using the PyTorch framework and LSTMs

After the project defense, I decided to switch from Scikit-Learn to PyTorch for developping and training my models. There was of course a pedagogical reason (*i.e.* learn the framework) but also a scientific one: in the scarce literature around PV/Load disaggregation and forecasting, *long short-term memory networks* are often used since they were built for problems with a temporal aspect. Thus, in order to try LSTMs and improve the performance of case 3, I decided to get familiar with PyTorch. I am greatly indebted to Daniel Godoy's [book](https://leanpub.com/pytorch) on the subject for this, and my own PyTorch *supra-framework* is based on his (especially the `Trainer` class). The library called `distorch` is the result of my learning - and should be working fine, nevertheless not error-proof and complete.

The `disaggregation.ipynb` notebook tries to more or less reproduce the experiments from phase 1 with the PyTorch/**distorch** framework.

**DO NOT consider the state of the notebook as a finished work. Outputs were cleared.**

## Using the disaggregated values

Traditional load forecasting models were built upon historical profiles presenting a well-known/ well-defined *load curve*. Nevertheless high-penetration of renewable/green energy on the grid severly alters the shapes of individual and even collective load curves. This is a serious challenge to DSOs and their actual models. Do we need to throw away these models or do we only need to disaggregate the net load and then use the obtained load values and traditional models?

This was the main question that my supervisors and I wanted to address, and the notebooks here present the first steps in that direction. Unfortunately, I was required elsewhere and the work here is now dormant somehow. Discussions to reframe the goals and steps should be held in order to revive this aspect of the project. The work would probably need to be restarted while recycling some code sections from the notebooks.

# Try the code

A public Docker containter is available to try the code:

```bash
docker pull pasokon2dev/internship.2021:extension1.0
docker run -p 8888:8888 pasokon2dev/internship.2021:extension1.0 
```

You can also clone the repository, but then need to set up a few environment variables using your CLI:

```bash
$<project/root/directory/>~> export PYTHONPATH=$PWD/lib
$<project/root/directory/>~> export ROOT_DIR=$PWD/work/post_defense_work
$<project/root/directory/>~> export DATA_DIR=$PWD/data/phase02
$<project/root/directory/>~> export MOD_DIR=$ROOT_DIR/saved_models
$<project/root/directory/>~> export RES_DIR=$ROOT_DIR/results
```

and install the python librairies:
```bash
pip3 install -r requirements.txt
```