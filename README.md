# Horse-Race-Predictor
Predicts the order of a horserace, given input data. 
The data tables and model objects have been omitted as they belong to https://www.betturtle.com/. 

Repo draws on book Hands-On Gradient Boosting with XGBoost and scikit-learn and some code shared on kaggle. 

Split into 3 types of model, the classification model isn't particularly well suited but is still used. 
5 Different model objects are used in a final ensemble notebook, with the XGRegressor model taking precedence. 

Most of the work went into the preprocessing.py file. With most of the logic behind it done outside of python (hence data insights is very limited).

For reference the raw data into the model should have ~200 columns which should be seen in preprocessing. 

Overall the model is ~3 positions out on average and has a return on investment of around -5% based on validation data, for betting on P1 (compared with -30% for favourites).
