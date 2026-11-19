# Delivery Performance, Review Timing, and Customer Feedback in Brazilian E-Commerce

**Master's Research Project | Social Media and Web Analytics | Ca' Foscari University of Venice**

## Project Overview

This project investigates how delivery performance, delay severity, and review submission timing are associated with customer ratings and delivery-related expressions in Brazilian e-commerce.

Using the Olist Brazilian E-Commerce Dataset, the study combines operational delivery records, numerical ratings, and Portuguese customer reviews to examine how customer feedback varies across delivery and review-timing groups.

The analysis also highlights why review timing and selection bias matter when interpreting customer satisfaction data.

## Research Question

*How do customer ratings and delivery-related expressions vary across delivery performance, delay severity, and review timing in Brazilian e-commerce?*

## Dataset

**Source:** [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

The final analytical sample includes:

- **96,353 scored review records** from 95,824 orders.
- **39,068 written-review records** from 38,931 orders.
- Historical Brazilian e-commerce transactions from 2016–2018.

The study combines order delivery information, customer ratings, and written feedback.

## Methods and Tools

**Programming:** Python

**Analytical methods:**
- Data cleaning, preprocessing, and dataset integration.
- Exploratory data analysis and descriptive statistics.
- Delivery delay and review timing classification.
- Rule-based Portuguese text analysis.
- Negative-rating analysis and 95% Wilson confidence intervals.
- Supplementary logistic regression and sensitivity analysis.

## Key Findings

**1. Delivery delays are strongly associated with negative customer ratings.**

The proportion of negative ratings increases from 9.28% for on-time deliveries to 79.19% for deliveries delayed by eight or more days.

**2. Review timing changes how delivery performance should be interpreted.**

Reviews associated with longer delivery delays were disproportionately submitted before customers received their orders. When considering only reviews submitted on or after delivery, the negative-rating gradient is substantially less pronounced.

**3. Written reviews reveal a different distribution of customer experiences.**

Customers who write comments do not necessarily represent the overall population of customers who provide ratings. This distinction is important when interpreting textual feedback.

**4. Text analysis identifies delivery-related concerns.**

Rule-based Portuguese expression matching identifies explicit references to delivery delays and non-receipt in customer comments. These detections are exploratory indicators, not independently validated complaint labels.

## Research Outputs

The repository includes:

- **Python source code:** Complete analytical workflow.
- **Research paper:** Academic report covering the research question, methodology, findings, and limitations.
- **Figures:** Three visualizations presenting rating distributions, delivery delay severity, and delivery-related expressions.

## Limitations

This research is observational and descriptive. The reported relationships do not establish causality.

Written reviews are a selected subset of scored reviews, and the rule-based text classification has not been independently validated against manually annotated customer feedback.

## Data Availability

The original Olist dataset is publicly available on Kaggle through the link provided above.

The raw dataset is not redistributed in this repository.

## Author

**Dan Vy Le**

Master's in Data Analytics for Business and Society

Ca' Foscari University of Venice, Italy

*Academic research project — Social Media and Web Analytics.*
