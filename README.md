# Gradient Ascent Final Project: Identifying Fake News Using Convolution and RNNs

Group members:
* Ariel Rotter-Aboyoun (arottera)
* Daniel Kostovetsky (dkostove)
* Julius Sun (jsun6)
* Raymond Cao (rcao6)

This project is a reimplementation of the work in [*Fake News Identification on Twitter with Hybrid CNN and RNN Models*](https://arxiv.org/pdf/1806.11316.pdf) (2018).

In order to run this code, go to https://github.com/thiagorainmaker77/liar_dataset to download the LIAR dataset and put these into a folder labeled 'data'.

## Environment Setup

This project requires Python 3.6 or newer.

To set up the virtualenv, make sure that `virtualenv` is installed (if not, then `pip install` it), then run
```bash
virtualenv -p /path/to/corrct/version/of/python venv
```

To install the necessary packages into this virtualenv, run
```bash
pip install -r requirements.txt
```

Finally, to activate the virtualenv, run
```bash
source venv/bin/activate
```
