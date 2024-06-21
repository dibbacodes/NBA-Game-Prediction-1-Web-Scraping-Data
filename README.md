# NBA Game Winner Prediction

## Overview

This project aims to predict the outcomes of NBA games using a machine learning model. It involves scraping game data, processing it, and applying a Ridge Classifier to make predictions. The project is divided into three main parts:

1. **Web Scraping and Data Collection**: Utilizing Playwright and BeautifulSoup to gather and parse historical game data from basketball-reference.com.
2. **Data Processing and Feature Extraction**: Parsing the collected HTML data to extract relevant features and creating a dataset suitable for machine learning.
3. **Machine Learning Model**: Building and evaluating a Ridge Classifier model to predict game outcomes, including improvements with rolling averages for better accuracy.

## Project Structure

### Part 1: Web Scraping with Playwright and BeautifulSoup

- **Objective**: Collect detailed data on NBA games, including team statistics, player performance, and historical game outcomes.
- **Tools**: 
  - Playwright: For automated browser interaction and data extraction.
  - BeautifulSoup: For parsing the HTML content.
- **Files**: 
  - `scrape_nba_data.py`: Script for scraping game and standings data from basketball-reference.com.
  - **Data Directories**:
    - `data/standings`: Stores the standings pages.
    - `data/scores`: Stores individual game box scores.

### Part 2: Data Parsing and Preparation

- **Objective**: Convert raw HTML data into a structured format suitable for analysis.
- **Tools**: 
  - Pandas: For data manipulation and analysis.
  - BeautifulSoup: For parsing the HTML content.
- **Files**:
  - `parse_nba_data.py`: Script for parsing HTML files and extracting game statistics into a DataFrame.
  - `nba_games.csv`: Combined dataset of parsed game statistics.

### Part 3: Building a Ridge Classifier Model

- **Objective**: Develop a machine learning model to predict NBA game winners.
- **Tools**: 
  - Scikit-learn: For building and evaluating the Ridge Classifier model.
  - Pandas: For data manipulation.
- **Files**:
  - `nba_model.py`: Script for training and evaluating the Ridge Classifier model, including feature selection and backtesting.
  - `nba_games.csv`: Used as input data for the model.

## Key Steps

1. **Scrape NBA Data**: Collect game data from basketball-reference.com for seasons ranging from 2016 to 2023.
2. **Parse and Prepare Data**: Process the HTML files to extract team and player statistics, and combine them into a structured dataset.
3. **Build and Evaluate Model**: Train a Ridge Classifier model to predict game outcomes, and improve the model using rolling averages of the last 10 games.

## Results

- **Initial Accuracy**: Baseline accuracy of the Ridge Classifier model.
- **Final Accuracy**: Improved accuracy after incorporating rolling averages.
- **Home Win Percentage**: Analysis of the impact of home advantage on game outcomes.

## How to Use

1. **Setup Environment**:
   - Install required Python packages using `pip install -r requirements.txt`.
2. **Run Data Collection**:
   - Execute `scrape_nba_data.py` to collect and save game data.
3. **Parse Data**:
   - Execute `parse_nba_data.py` to convert HTML files into a structured dataset.
4. **Train and Evaluate Model**:
   - Execute `nba_model.py` to train the model and evaluate its performance.

## Requirements

- Python 3.8+
- Required Python packages: BeautifulSoup, Playwright, Pandas, Scikit-learn

## Conclusion

This project demonstrates the process of using web scraping, data parsing, and machine learning to predict NBA game outcomes. The use of rolling averages and feature selection techniques helps in improving the model's accuracy, providing valuable insights for sports analytics.

