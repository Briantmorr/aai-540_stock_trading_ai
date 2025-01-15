# Stock Market Prediction with MLOps Pipeline

## Project Overview
This project focuses on predicting the stock prices of FAANG (Facebook, Amazon, Apple, Netflix, Google) companies using historical stock market data. The primary objective is to design and implement an MLOps pipeline that supports continuous learning, ensuring the model stays relevant and accurate in the face of ever-changing market conditions.

While the stock market's volatility makes prediction challenging, this project leverages MLOps principles to automate model updates, monitor performance, and detect data drift. Although real-time trading and advanced features like sentiment analysis are not within the immediate scope of this project, these are exciting areas we aim to explore in future iterations.

---

## Project Goals
1. **Develop a Stock Prediction Model**
   - Focus on predicting the next day's closing prices for FAANG companies.
   - Simplify the feature set to ensure clarity and efficiency in the first iteration.

2. **Build a Continuous Learning Pipeline**
   - Automate model retraining and redeployment in response to market shifts.
   - Include monitoring for performance degradation and data drift.

3. **Explore Real-World Applications**
   - Set a foundation for future work in real-time trading and advanced analytics.
   - Develop strategies for A/B testing and paper trading in subsequent projects.

---

## Dataset
We use the **FAANG-Complete Stock Data** dataset from Kaggle ([Dataset Link](https://www.kaggle.com/datasets/aayushmishra1512/faang-complete-stock-data)). This dataset includes:

- **Features**: Date, Open, High, Low, Close, Adj Close, Volume
- **Scope**: Historical stock data for FAANG companies.

For this project, we will primarily focus on:
- **Date**: As a time index for predictions.
- **Open**: The stock's opening price.
- **Volume**: The total volume of trades.

Other features such as High, Low, and Adj Close are omitted in the initial implementation due to potential multicollinearity.

---

## Key Components
1. **Stock Prediction Model**:
   - Supervised regression model predicting the next day's closing price.
   - Evaluation metrics: Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

2. **MLOps Pipeline**:
   - **Performance Monitoring**: Track MAE and RMSE to ensure accuracy.
   - **Data Drift Detection**: Identify changes in input data distributions that could impact predictions.
   - **Automated Alerts**: Notify stakeholders of significant performance degradation or unusual data patterns.

3. **Future Directions**:
   - Integrate real-time trading mechanisms.
   - Incorporate sentiment analysis to capture market emotions.
   - Perform A/B testing of various feature sets and strategies using paper trading and back-testing.

---

## Scope for This Class
The primary focus is to:
- Deliver a simplified yet robust **MLOps pipeline**.
- Ensure the model remains up-to-date and adaptable to market changes.
- Lay the groundwork for more advanced features and strategies in future projects, such as real-time trading and LLM-based sentiment analysis.

**Stretch Goals**:
- Implement a basic paper trading module to simulate trading strategies.
- Explore lightweight feature engineering techniques to enhance the model without adding significant complexity.

---

## Future Vision
This project is part of a larger journey to create an advanced AI-driven stock trading system. By starting with a strong foundation in MLOps, we aim to:
- Build scalable and maintainable pipelines.
- Experiment with advanced predictive features like sentiment analysis and algorithmic trading strategies.
- Develop a comprehensive trading ecosystem that includes real-time data aggregation, decision-making, and trade execution.