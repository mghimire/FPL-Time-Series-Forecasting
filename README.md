# FPL Time Series Forecasting
## Introduction

The **Fantasy Premier League (FPL)** is an online fantasy football game based on the **English Premier League (EPL)**. Players build teams of players from the EPL and compete against each other in pursuit of the highest accumulated score over the course of a league season.

A fantasy player receives a score based on the performances of the individual players on their team per **Gameweek (GW)**. This performance is delineated in the form of points, which can be earned or lost via various game statistics such as goals, assists, and saves. These points per statistic are awarded differently for players of different positions. Details on the point rubric can be found under the scoring tab on the [FPL Website](https://fantasy.premierleague.com/help/rules). There is also further information there for those who may be interested in learning more about the league or participating in the game.

The objective of this project is to use **statistical** and **machine learning (ML) tools** to try and forecast the performance of FPL players.

### Disclaimer:
Although I am an avid enjoyer of FPL and the EPL in general, I do not endorse the game or the league in any capacity. This repository is intended to be a purely academic exploration. I do not take any responsibility for the use and results of the tools presented in this project. However, if you find any success with this model, please feel free to share. Conversely, if you find anomalous results or mistakes in the code, please notify me. Collaborations are welcome and adjacent exploration is highly encouraged.

## Theoretical Outline

This study is based on **Time Series Forecasting** and **ML techniques** surrounding the subject. **Non-ML statistical tools** such as **autoregressive integrated moving average (ARIMA)** and **seasonal decomposition of time series (STL)** may also be used at times in feature-engineering or cross-examining the performance of the **ML techniques**. However, the primary goal of this study is to use **ML techniques**, such as **recurrent neural networks (RNNs)** and **long short-term memory networks (LSTMs)**, to try and optimize accuracy and consistency in predicting FPL player scores.

The model will be constructed to predict the score of a player on any given **GW**. It will only work for one player at a time<!--, but depending on the desired complexity and available computational power, neural networks built for one player may be incorporated in the neural network for another player-->. Therefore, in order to build an "optimal" team, one would have to run the model on several different players and forecast for multiple future weeks. There is also the option to increase the complexity of the architecture based on the available computational power in order to potentially improve the predictions. This would include incorporating data for the player's team's performances, matchups against particular opposing teams, matchups against particular opposing players, and (in an ideal world) the performances of other players in the league (including relevant injuries and suspensions). However, due to the increase in computational load, certain data would have to be prioritized when adding newer features. This will be discussed in the **Feature Engineering** section.

The training data will be in the **Time Series Format**. This will be represented as a **Pandas DataFrame** with the **time values (dates)** representing the **keys (indices)** and the **columns** encompassing various **features** based on the desired complexity of the **ML architecture**. The default and minimally required feature will be the score of the player in question. The test data will simply be the score of the player in past GWs. Training data will be used to train a **neural network (NN)**, forward propagating feature information through hidden layers to make predictions. Test data will be used to determine the accuracy of the predictions, which will then be used to modify the internal parameters of the **NN** using back propagation. This will be repeated with all the available training data, going back to four seasons in the EPL.

### Feature Engineering



## Data Acquisiton

## Model Features and Parameters

## Performance

## Conclusions and Ongoing Work
