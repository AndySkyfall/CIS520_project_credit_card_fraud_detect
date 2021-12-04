# CIS520_project_credit_card_fraud_detect
# Credit Card Fraud ML Detector           
# last update - 11.27 by Yuxuan Hong                                                                                                              

# Data- Use this website to download the dataset
https://www.kaggle.com/mlg-ulb/creditcardfraud#creditcard.csv

# How to use - CLI
- To test adaboost with different thershold, run `python3 adaboost.py -d creditcard.csv -n 10 -t 0.5`
- To test adaboost with different upsample rate, `run python3 adaboost.py -d creditcard.csv -s 1 -n 10`
- To test fc_nn with different upsample rate, run python3 fc.py -d creditcard.csv -s 1 -n 10
- To test svm with different upsample rate, run python3 svm.py -d creditcard.csv -s 1 -n 10
- To test any model with just an upsample ratio but not a range just use "-n". For example, python3 svm.py -d creditcard.csv -n 10
- The upsample range will be [1, n]
- To automatically generate curve on three models, run python3 run_pipeline.py -d creditcard.csv -s 1 -n 10
