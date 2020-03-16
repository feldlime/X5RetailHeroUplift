RetailHero Uplift Modeling
==========================

# About
Задача на uplift-моделирование. Необходимо отранжировать клиентов по убыванию эффективности коммуникации.

Страница соревнования: https://retailhero.ai/c/uplift_modeling/

Data:
- data/clients.csv
- data/products.csv
- data/purchases.csv — история покупок клиентов до смс кампании
- data/uplift_train.csv — обучающая выборка клиентов, информация о коммуникации и реакции
- data/uplift_test.csv — тестовые клиенты, для которых необходимо оценить uplift

## Final score

*Public: #7 place 0,1060*
*Private: #4 place 0,098330*

[Used features](https://github.com/feldlime/X5RetailHeroUplift/wiki/Features)

# Run

`python main.py`

# License

This project is open-sourced software licensed under the MIT license.
