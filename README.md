# DACON Fake News Identification

[Competition Link](https://dacon.io/competitions/official/235658/overview/)

### Description 
Given a data of the news, we classify if the news contains real or fake information. 

### Exploratory Data Analysis 
[Refer to this notebook](). We look at the provided data frames and try to do some preliminary analysis between the features. 

### Rule Based Model 
[Simple rule based submission without any machine learning](https://github.com/puzzlecollector/DACON-fake-news-identification/blob/main/Rule%20Based%20Model.ipynb). Scores an accuracy of 95.6%. This is the benchmark score that (whatever ML model we use) should surpass.If the content in the test data also exists in the train data, assume that this news is fake, otherwise assume that it is not fake.  

### Bidirectional LSTM Model 
[Data Processing using konlpy Mecab and keras Tokenizer]() 
[Training a Bidirectional LSTM model with title and content inputs](https://github.com/puzzlecollector/DACON-fake-news-identification/blob/main/fintech_nlp.ipynb)
