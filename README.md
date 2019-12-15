# Gradient Ascent Final Project: Identifying Fake News Using Convolution and RNNs

Group members:
* Ariel Rotter-Aboyoun (arottera)
* Daniel Kostovetsky (dkostove)
* Julius Sun (jsun6)
* Raymond Cao (rcao6)

This project is a reimplementation of the work in [*Fake News Identification on Twitter with Hybrid CNN and RNN Models*](https://arxiv.org/pdf/1806.11316.pdf) (2018).

The contents of the `data` directory were downloaded from https://github.com/thiagorainmaker77/liar_dataset.

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

## Running the Models
This project contains three models:
- LSTM: a standard model conaining an LSTM
- LSTM with Dropbout: a model similar to the LSTM model, but with a dropout layer added
- Hybrid: a model containing an LSTM component and a 1D CNN with maxpooling

To run any code, navigate to the `code` directory.

To train and see test result for all three models from the paper, run:
```bash
python main.py
```

To input a custom sentence for the hybrid model to analyze, run:
```bash
python main.py --test-sentence "This is test sentence"
```
