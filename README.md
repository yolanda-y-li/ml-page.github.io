# ML Group 45 Github Page


MIDTERM REPORT 

<img width="1272" alt="Screenshot 2024-04-02 at 10 48 00 PM" src="https://github.com/vinwu2/ml-group45-sp2024/assets/161073357/4cabbadc-39c7-4caa-9469-d30dddf93586">

<img width="674" alt="Screenshot 2024-04-02 at 10 48 39 PM" src="https://github.com/vinwu2/ml-group45-sp2024/assets/161073357/17c89ec1-d4c6-471f-bc48-8c6b6c57ed77">


Section 1: Introduction/Background

Several studies have experimented with using machine learning to make more accurate NBA predictions regarding sports betting. [2] uses a probabilistic model with data on individual player’s history to calculate expected points per possession. However, most studies use team-level statistics. [3] was able to achieve an overall 70% accuracy with linear regression, logistic regression, support vector machines, and artificial neural networks using team-level data, with models that combined predictions from multiple sources producing the best results. [1] combined team-level statistics with clustering analysis of players based on playing style to achieve over 70% prediction accuracy. Consequently, based on these results, we aimed to refine our model to focus on forecasting the spread for NBA games. The term spread simply refers to placing a bet on the total projected points contributed by both teams given a line. Oftentimes, users can choose to bet “over” or “under” the projected point total. As this value is binary, we believed that analyzing historical team-data would produce more consistent results, thus allowing our model to have higher accuracy. Some of the different factors we have to take into account for this model include: home versus away performance, head-to-head matchups, win-loss records, and player injuries. We intend to constrict our data to the most recent 2-3 years to accommodate roster modifications and performance trends with the current coaching staff. 

Dataset Information:

Home vs Away Data, Points Per Game, Rebounds Per Game, Shot Percentage, Assists, Game Records: https://www.kaggle.com/datasets/nathanlauga/nba-games

Individual Player Data: https://www.kaggle.com/datasets/justinas/nba-players-data/data

Team Records: https://data.world/gmoney/nba-team-records-by-year

Current Player Injuries: https://www.kaggle.com/datasets/ghopkins/nba-injuries-2010-2018

NBA Team Level Statistics (FG%, 3P%, FT%, ORB, DRB, AST, STL, TOV, BLK) Rankings per Season Summary: https://www.basketball-reference.com/leagues/NBA_2023.html

Section 2: Problem and Motivation

The amount of uncertainty involved in NBA games makes betting outcomes difficult to predict. We noted that individual player betting propositions are influenced by many factors: performance, subtle injuries, location, psychological state, etc., which are often hard to reliably predict as there is very limited public data. Additionally, less implicit features, such as back-to-back games, travel distance, and specific matchups all impact player performance which can produce inconsistent results. It is also important to note that sports betting applications are volatile in the sense that odds change quickly throughout the day as they adjust to incorporate new information. This leaves very limited opportunities to place a high value bet on an individual player. Due to these reasons, our group decided to redirect our focus on team-level betting, as there are less variables to account for. 

Our motivation is to enhance the accuracy of NBA betting predictions. Using historical data and machine learning models, we aim to predict individual game outcomes with relatively high accuracy. In return, individuals may potentially witness financial rewards. In cases where our machine learning model predicts the underdog to win, the payout for betting on the underdog team will increase. Moreover, many people enjoy sports betting due to the additional thrill and engagement it adds when watching a game. Betting on the spread is relatively straightforward and adds an extra layer of interest for users.


Section 3: Methods 

Data Preprocessing Method:  

The first dataset we looked at contained NBA statistics for each season. They were individual tables, covering data from 2020-2023. We had to clean and alter these tables in a few ways. The first thing we did was to remove the unnecessary columns, such as games played and minutes played (as each team played a nearly equivalent number of games and minutes when we did not consider playoff games). We then removed the row that covered the sum of stats across all teams in the NBA, as we want to differentiate between teams, not treat them as one group. We then renamed a few columns that had confusing titles. We did this for each of the four tables. Finally, we combined these tables and sorted them by team. This means that every four rows covered a different team, with each row in each group of four covering the data for a team for the respective year. As such, we would have important seasonal statistics for each team over a course of four years, providing multiple features and observations. A lot of this was done by hand, rather than by code.

<img width="625" alt="Screenshot 2024-04-02 at 10 49 54 PM" src="https://github.com/vinwu2/ml-group45-sp2024/assets/161073357/720ef469-7fe2-4c1f-87a2-24b680d74a05">

The next dataset we used was more complicated and much larger.  It contained two important tables: the first contained data for each game played in the NBA from 2004-2020, which is understandably a very large number of games. It divided this data into home and away teams, along with the general performance for the home team vs away team. This provided important data on how each team performs when playing at home vs playing away from home. However, there were a few problems with this dataset. The first was that rather than having the team name, it utilized the NBA team ID, which was much more difficult to work with. This brings us to the second table. The second table mapped each NBA team ID to the team nickname. As such, we decided to merge the two tables based on their team ID. Before doing this, we deleted unnecessary columns in this second table, only keeping the team ID and nickname.We also decided to divide the table into two dataframes, with one containing home data for each team and the other containing away. This meant that for the first dataframe, we merged based off Away Team ID in the first table to the NBA team ID in the second table, giving us the nickname of each away team and allowing us to analyze how each team plays away from home.
For the second dataframe, we merged based off Home Team ID to the NBA team ID, giving us the nickname of the home teams and allowing us to analyze how each team performs at home. Overall, both tables kept home and away game data (such as 2 point%, total points, etc.) so we could analyze both how many points a team scored, but also how many they allowed. We also sorted both tables based on their respective nicknames.

<img width="629" alt="Screenshot 2024-04-02 at 10 50 25 PM" src="https://github.com/vinwu2/ml-group45-sp2024/assets/161073357/93dd07ea-adc0-4796-8dd9-73fee42c4e71">

<img width="624" alt="Screenshot 2024-04-02 at 10 50 37 PM" src="https://github.com/vinwu2/ml-group45-sp2024/assets/161073357/3f57d92d-d3f4-4059-844d-b7906533e41d">

<img width="626" alt="Screenshot 2024-04-02 at 10 50 49 PM" src="https://github.com/vinwu2/ml-group45-sp2024/assets/161073357/44d6e061-d95a-48ea-b194-13cb941457e2">

Since there was too much data, we also added the GAME_DATE_EST column back and cut down on all games before 2018. This both lessened the data our model needed to go through, while also removing data that is highly irrelevant to current team performances. 

We then preprocessed the data again so now for each game there are rolling averages computed for each of the main statistics as well as the percent change over the last 7 days. Our newly processed datafiles can be accessed through Github.

ML Algorithm/Model & Supervised Learning Method:

Logistic Regression Supervised Learning Method:

<img width="586" alt="Screenshot 2024-04-02 at 11 23 33 PM" src="https://github.com/yolanda-y-li/ml-page.github.io/assets/142852630/e560336b-3db0-46ec-a9e9-3800952d5bbd">

<img width="590" alt="Screenshot 2024-04-02 at 11 24 11 PM" src="https://github.com/yolanda-y-li/ml-page.github.io/assets/142852630/be4e585f-5f26-4971-aa9a-579a128f41a1">

<img width="576" alt="Screenshot 2024-04-02 at 11 24 29 PM" src="https://github.com/yolanda-y-li/ml-page.github.io/assets/142852630/20f78bc1-af2f-4424-a5d9-fac14ac666fc">

<img width="469" alt="Screenshot 2024-04-02 at 11 24 58 PM" src="https://github.com/yolanda-y-li/ml-page.github.io/assets/142852630/35ad498e-b8b7-4dfc-ab64-0a95b7c4d60a">

<img width="497" alt="Screenshot 2024-04-02 at 11 25 13 PM" src="https://github.com/yolanda-y-li/ml-page.github.io/assets/142852630/bf552971-0c38-4e95-a1d4-6fcf4bf3e22a">

We used logistic regression to predict if we can forecast the next game going over the betting over / under total on points. To do this, the features mentioned previously (FG%, 3P%, FT%, ORB, DRB, AST, STL, etc.) were fed into a logistic regression algorithm. We chose to use a logistic regression algorithm because it allows us to perform binary classification. “Over” and “under” are the two possible results we can have, so binary classification is suitable here. Additionally, logistic regression provides easily interpretable results; the coefficients represent the impact of each feature on the odds of the outcome, making it easy to understand the relationship between over / under outcomes and our independent variables. It is also not as susceptible to overfitting, and works well with small and large datasets (it is fairly simple and not very computationally expensive). In addition, the logistic regression model can be considered as a supervised learning method as the input-output pairs are correlated. Our file containing the data and results for the linear regression model can be found on the Github repository. 

Section 4: Potential Results and Discussion

Visualizations of Results: 

Figure 1: ROC chart showing a flow of true positives versus false positives. We can measure the area under the curve to determine the total accuracy. 

<img width="436" alt="Screenshot 2024-04-02 at 11 26 05 PM" src="https://github.com/yolanda-y-li/ml-page.github.io/assets/142852630/f07235a4-5f78-445e-b930-891601cb9f16">

Figure 2: PCA Component analysis split into 2 components for class analysis. 

<img width="403" alt="Screenshot 2024-04-02 at 11 26 36 PM" src="https://github.com/yolanda-y-li/ml-page.github.io/assets/142852630/e24a7dcb-aa00-4d22-9fb6-d29bbf560daa">

Quantitative Metrics:

To determine the accuracy of our linear regression model, we can utilize several quantitative metric techniques. Previously, in the project proposal, we determined that we were going to use balanced accuracy, brier score, and mean squared error as a metric of determining the level of our error. However, since we are only partially done with our implementation, we will use simpler quantitative metrics for now.

So far, we have implemented a logistic regression model that categorizes each game as over / under, treated as 1 / 0 respectively. We can analyze the results of our logistic regression model using precision score, recall score, F1 score, and more, all included in the classification report:

Precision = # true positives / # predicted positives

Recall =  # true positives / # actual positives

F1 score = 2 * Precision * Recall / (Precision + Recall)

Figure 3: Classification Report

<img width="437" alt="Screenshot 2024-04-02 at 11 27 40 PM" src="https://github.com/yolanda-y-li/ml-page.github.io/assets/142852630/6f2a47b6-1f5f-4088-a1ee-eab9173e8612">

The screenshot above depicts the report of the three quantitative measures: precision, recall, and f1-score. The accuracy of our data is reported as well, implying that our model is 58% correct. It is evident that changes needed to be made regarding our implementing procedures. This could be done by removing misleading variables. The quantitative metrics on the whole provide us with further understanding as it explains the performance of our model. Through the numerical values provided, we can conduct multiple trials to increase the overall efficiency. 

Analysis of ML Algorithm/Model:

Based on the visualizations and the values obtained in the quantitative metrics subsection, it is evident that our current model has room for improvement. As we are dealing with sports betting applications and monetary value is involved, we want to maximize the accuracy of our model so that user’s aren’t losing any money. Although, to obtain these results, we implemented a logistic regression supervised learning method to accurately predict whether a game will go over / under the betting total. Our model consists of various independent variables (FG%, 3P%, FT%, ORB, DRB, AST, STL, etc.) that provide a threshold of projected points. Using these independent variables, we can calculate the dependent variable (how many points a team will score). Based on these values, we can compare projected points and determine which team will win the game. Revisiting our model that displayed this information, we can see based on the visualizations that there is evident pattern within the data. For instance, there is a general upward trend over time for the ROC curve. This likely means that the accuracy within our model is likely improving.  Additionally, the quantitative metrics allow us to see a representation of the error within our model. Due to the level of error being 58%, we can try to identify the rationale behind why these errors are existing. Oftentimes, this may be due to an irregular game where individual players perform poorly arbitrarily. In addition, we can compare our obtained values using the linear regression method to other existing machine learning models to determine our level of accuracy. In a study conducted by Harvard graduates, a PCA analysis was conducted on a dataset consisting of NBA team level stats over a decade to predict conference ranking for the upcoming season. After conducting the PCA test, only several key basketball statistics (FG%, TV, OR, DR) were highlighted. After making revisions to the model through multiple trials, the final forecast that was conducted by the students attained a mean accuracy score error of 19% indicating there was minimal error.  Based on these results, it is apparent there are independent variables influencing our model that are not accurate representations of how well a team may perform. We must identify and remove these excess independent variables to receive more accurate results as desired. 

Next Steps:

Based on the results obtained for the midterm checkpoint, some of the next steps that we need to prioritize include ensuring that the predicted values from our machine learning model match up with actual values. To minimize error, we need to identify potential variables that are negatively impacting our data. Additionally, when comparing our model with other studies regarding NBA sports betting, we found that most models referred to the PCA method to determine spread. Our first step to do this is by standardizing each individual data point so that the mean is equal to 0 and the standard deviation is equal to 1. Some of the other potential steps we can follow prior to the Final report deadline include applying regularization techniques such as Ridge Regression to prevent any overfitting patterns within the data. Furthermore, to add additional security measures, an idea that our group hypothesized to increase overall machine learning model accuracy is to cross-validate the performance to obtain more reliable estimations of predictive accuracy. Although, it is important to note that there are days in any sport in which underdogs may win due to unforeseen circumstances. Therefore, we must note that having some error is inevitable, but we are aiming to minimize the chances of such error. Through the next couple of steps, our group will implement various normalization and regularization techniques to prevent overfitting and identify potential gaps that exist within our current model. 
 

Section 5: References (IEEE Format)

[1] C. Osken and C. Onay, “Predicting the winning team in basketball: A novel approach,” Heliyon, vol. 8, no. 12, p. e12189, Dec. 2022, doi: https://doi.org/10.1016/j.heliyon.2022.e12189. 

[2] J. Kuehn, “Accounting for Complementary Skill Sets: Evaluating Individual Marginal Value to a Team in the National Basketball Association,” Economic Inquiry, vol. 55, no. 3, pp. 1556–1578, Apr. 2017, doi: https://doi.org/10.1111/ecin.12451. 

[3] M. Beckler, H. Wang, and M. Papamichael, “NBA Oracle,” 2013. Available: https://www.mbeckler.org/coursework/2008-2009/10701_report.pdf 


