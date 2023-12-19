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

### Starting Model

We will start by creating a rough predictive model based on the FPL scoring system. This will be based on the point rubric as well as a baseline understanding of the sport of football.

At the **zeroth order**, we have the points players get for simply playing: 1 point for < 60 mins of play and 2 points otherwise. This can be calculated using a simple expectation value based on the probability that a player plays < 60 mins or otherwise, which should be **weighted with a recency bias (WWRB)**. Note that we will use recency bias for almost all the statistics we use because we expect more recent data to be more relevant than older data in determining trends.

The best predicter of future success is past and present success. Therefore, at the **first order** we have the average performance of the player **WWRB**. This performance can be gleaned from game statistics such as goals, assists, saves, clean sheets, goals conceded, and yellow/red cards as relevant for particular player positions. Some of these statistics (goals, assists, yellow/red cards) should be converted into per-minute units and multiplied by the expected number of minutes played to estimate their contributions. Clean sheets, saves, and goals conceded are more dependent on opposition strength and team success (or lack thereof). The **zeroth** and **first order** contributions should be the minimal working parts of any model. Further improvements will increase accuracy but at an increasing computational cost.

The **second order** contribution for a player should be team success. This is particularly relevant for defenders and goalkeepers, because clean sheets and goals conceded are team-wide statistics that impact all the defensive players' scores. Clean sheets impact goalkeepers, defenders, and midfielders, and they also contribute a different number of points to each position, so this impact is calculated based on general team performance **WWRB**. Sometimes teams go through exceptional spells of defensive cohesion or dreadful lulls, so the primary contributing factor here would be the team's defensive trends as to the number of goals conceded.

The **third order** contribution for a player would be the particular opposing matchup. Some players struggle against particular teams or particular opposing players, and this should introduce a higher-order correction in predicting the player's success, but not as much as recent individual and team success.

The **fourth order** contribution (arguably tied for third order) should be from the home or away status of games. This affects some players and teams more than others, and this factor should be gleaned from the difference in average statistics between home and away games per player per team.

Finally, the **fifth order** contribution would be from the performances of teammates and opposing players in the particular games. Given the computational power, one could simulate the indvidual stats of all players involved in the game with particular weights based on some or all of the aforementioned ordered factors, and these could be used to calculate the rough number of game stats for the particular player in question. For example, if we are trying to predict the performance of a Liverpool midfielder when playing against Manchester United, we would take into account his teammates' propensity to score in order to correct the player's assist numbers, their propensity to assist in order to correct the player's goal numbers, and the opposition's propensity to score in order to correct the player's clean sheet prediction. This would be a smaller order correction than the aforementioned categories, but could be relevant in particular polarizing matchups.

These predictive steps can be applied in a regressive manner with specified weights and biases for recency to reach a desired accuracy/consistecy, or in a **ML** context where all these contributions are given particular weights and biases that can be trained. We will talk more about this in the **Model Features and Parameters** section.

## Data Acquisiton

FPL Data has graciously been uploaded on Github by [Vaastav Aanand](https://github.com/vaastav) on his [Fantasy Premier League Repository](https://github.com/vaastav/Fantasy-Premier-League). Please follow him and support his work; he seems like a cool guy.

The link above contains data as far back as the 2016-17 season, so we have plenty of training and testing data for the various models we employ. Of course due to the transfer market, there may be limited data available for certain players (read: Erling Haaland). There will be a script that downloads all data from the aforementioned repository to your local working directory.

## Model Features and Parameters

### Feature Engineering

### Choice of Layers

## Performance

## Conclusions and Ongoing Work
