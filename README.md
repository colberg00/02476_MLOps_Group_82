# 02476_MLOps_Group_82

README file for Group 82.

1. The dataset we will work with is: Jia555/cis519_news_urls. The short hash for the data we will use is: "6dca88d". 

2. The model we would like to work with is microsoft/deberta-v3-base. We will fine tune it for our classification task.

3.
The overall goal of the project is to predict the news source by the keywords in the news article URL.

We intend to use the full dataset to run on. It contains 40858 URLs, of which 19787 is from Fox News and 210721 is from NBC News. The data only contains a single column, which is the full URL from the original news article. The data is therefore text. The output we seek is categorical.

We will use logistic regression as our baseline model (combined with TF-IDF).