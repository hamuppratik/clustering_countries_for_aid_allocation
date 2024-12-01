# Clustering Countries For Aid Allocation

## Project Overview
HELP International, an NGO committed to alleviating poverty and providing humanitarian aid, has raised $10 million to allocate strategically to countries most in need. This project categorizes countries based on socio-economic and health indicators to help the NGO's leadership make data-driven decisions on aid distribution.

**Tableau Dashboard Link:** https://public.tableau.com/views/ClusteringCuntriesforAidAllocation/Dashboard1?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link


## Table of Contents
1. Problem Statement
2. Objective
3. Dataset Description
4. Exploratory Data Analysis (EDA)
5. Hypothesis Testing
6. Machine Learning Modelling
7. Insights and Recommendations
8. Model Deployment

 
## 1. Problem Statement
HELP International aims to identify and categorize countries based on socio-economic and health factors. This will allow targeted allocation of $10 million to maximize the impact of aid.

## 2. Objective
- Categorize countries using socio-economic and health indicators.
- Identify countries in dire need of aid.
- Provide strategic recommendations for aid allocation.


## 3. Dataset Description
The dataset includes the following features:

- Country: Name of the country.
- Child_mort: Death rate of children under 5 years per 1,000 live births.
- Exports, Imports, and Health: As percentages of GDP.
- Income: Net income per person.
- Inflation: Annual GDP growth rate.
- Life_expec: Life expectancy at birth.
- Total_fer: Fertility rate.
- Gdpp: GDP per capita.
- 
## 4. Exploratory Data Analysis (EDA)


- Checked for missing values and outliers. No missing values were found, and outliers were handled using Winsorization.
- **Feature Engineering:**
New features such as Exports per Capita, Imports per Capita, Health Spending, and Inflation-adjusted GDP per capita were created for better insights.
        Indicators like High Child Mortality and Low Life Expectancy were added for prioritization.
- **Scaling:**
Standard scaling was applied to normalize numerical features for better clustering performance.
- Visualized the data using histograms, box plots, polar plot and scatter plots to identify relationships between variables.
- Key correlations:
-       i. Child Mortality negatively correlates with Life Expectancy (-0.89).
       ii. Income positively correlates with GDP per capita (0.90).


## 5. Hypothesis Testing

1. Health Spending vs Life Expectancy
- Null Hypothesis (H₀): Health spending does not impact life expectancy.
- Result: Rejected H₀. A significant positive relationship was observed.
2. Fertility vs Income
- Null Hypothesis (H₀): No correlation exists.
- Result: Rejected H₀. A strong negative correlation (-0.74) was found.
3. Income vs Child Mortality
- Null Hypothesis (H₀): Income does not impact child mortality.
- Result: Rejected H₀. Income negatively correlates with child mortality (-0.87).
4. Inflation vs GDP per capita
- Null Hypothesis (H₀): No relationship exists.
- Result: Rejected H₀. A negative correlation was identified (-0.33).


## 6. Machine Learning Modelling

**Clustering Models**
1. K-Means Clustering:

- Chosen number of clusters: 3 (based on Elbow Method and Silhouette Score).
- Silhouette Score: 0.29.
- Provided interpretable clusters and computational efficiency.

2. Hierarchical Clustering:

- Number of clusters: 3.
- Used Dendrogram for visualization.

3. DBSCAN:

- Did not perform well due to the dataset's nature (Silhouette Score: -0.037).

### Final Model: K-Means
- Selected due to its simplicity, interpretability, and effectiveness.
- Used all features with standard scaling for better cluster formation.

**Feature Importance Calculation:**
- The feature importance was determined by analyzing the distances between the data points and their respective cluster centroids:
1. Calculated the Euclidean distances of data points to all cluster centroids.
2. Identified the closest centroid for each data point.
3. Measured the absolute differences between each feature's value and its corresponding centroid value.
4. Ranked features based on these differences and selected the top 3 most important features that significantly influenced the clustering process.

- Example:
For a specific data point, the top 3 features contributing to its cluster assignment could include:

- GDP per capita (gdpp)
- Child mortality (child_mort)
- Health spending (health_spending)

This analysis highlights which factors play the most critical role in clustering, providing actionable insights for resource allocation.


## 7. Insights and Recommendations
**Cluster Characteristics:**

- Cluster 1: Countries with low income and high child mortality (critical need for aid).
- Cluster 2: Countries with moderate development and socio-economic indicators.
- Cluster 3: Developed countries with high income and low child mortality.

**Recommendations:**

- Allocate a majority of funds to Cluster 1 to address urgent needs.
- Consider Cluster 2 for programs supporting development initiatives.


**Scores and Metrics**
- Final Clustering Silhouette Score: 0.29.
- Optimal number of clusters: 3 (based on data distribution and domain insights).


## 8. Model Deployment

The machine learning model was successfully deployed as a web application to provide cluster predictions and identify key factors contributing to cluster assignments. Below are the steps undertaken for deployment:

## Framework and Tools:
* Flask
* Pickle
* HTML Templates

## Functionality:
* **Home Page:**
  * Manual input or file upload
* **Manual Input Prediction:**
  * Feature engineering
  * Data scaling
  * Cluster prediction
  * Top 3 influential features
* **CSV File Upload:**
  * Batch processing
  * Cluster prediction
  * Top factors for each country

## Key Features:
* Feature Engineering
* Cluster Classification (3 classes):
  * Very Poor Class
  * Bourgeoisie Class
  * Very Rich or one of the Top Class
* Feature Importance Analysis

## Deployment Workflow:
1. **Input Handling:**
   * Manual input
   * File upload
2. **Data Preprocessing:**
   * Feature engineering
   * Data scaling
3. **Prediction and Output:**
   * Cluster assignments
   * Key influencing factors
4. **Error Handling:**
   * Clear error messages

## Testing:
* Postman API testing
* Manual input and file upload testing

The deployment ensures an intuitive and reliable platform for:
* Country cluster prediction
* Understanding socio-economic factors


# Website Preview:

![alt text](<Front End SS upload root.png>)