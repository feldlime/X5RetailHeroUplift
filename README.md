[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/rugleb/surname-detection/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/Python-3.6%20%7C%203.8-green)](https://www.python.org/)

RetailHero Uplift Modeling
==========================

## About

Uplift-prediction task. It's necessary to sort clients by decreasing communication efficiency.

The competition page: https://retailhero.ai/c/uplift_modeling.

Data:
- data/clients.csv
- data/products.csv
- data/purchases.csv - customer purchase history before sms campaign.
- data/uplift_train.csv - training sample of clients, information about communication and reaction.
- data/uplift_test.csv - test clients for which it is necessary to evaluate uplift.

## Final score

*Public: #7 place 0.1060*

*Private: #4 place 0.098330*

## Features

You can find description of all used features [here](https://github.com/feldlime/X5RetailHeroUplift/wiki/Features).

## Run

```
python main.py
```

## License

This project is open-sourced software licensed under the [MIT license](https://github.com/feldlime/X5RetailHeroUplift/blob/master/LICENSE).
