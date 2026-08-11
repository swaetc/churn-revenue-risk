# SA Telecom Churn Prediction with Revenue-at-Risk Ranking

I noticed a lot of public discussion (news articles, Hello Peter complaints,
price-war headlines) about SA mobile customers switching networks over pricing
and data bundle changes, especially with RICA number-porting making it easy to
switch without losing your number. I wanted to see whether churn is actually
predictable from usage patterns, and more importantly, whether it's the
high-value customers or the low-value ones driving the losses, since that
changes what retention should actually do about it.

## Business question
Which customers are about to churn, how much monthly revenue does each one
represent, and who should the retention team call first given they can only
call a limited number of people per week?

## Method
Base dataset is the widely-used Telco Customer Churn dataset, rescaled to
realistic South African postpaid mobile ARPU bands (R250-R900/month). Compared
a Logistic Regression baseline against XGBoost. Used SHAP for model
explainability. Built a revenue-at-risk ranking (churn probability x monthly
value) to prioritize retention calls by actual dollar exposure rather than
raw churn probability alone.

## Key results
- Logistic Regression and XGBoost performed comparably (ROC-AUC 0.841 vs
  0.840), suggesting the churn signal is largely linear and well captured by
  a handful of strong features
- At a decision threshold of 0.35: 88% recall, 46% precision on the churn
  class, a deliberate trade-off since missing a high-value churner costs more
  than a wasted retention call
- SHAP shows contract type, tenure, and monthly charges as the dominant churn
  drivers, with fiber optic service and electronic check payment as secondary
  but consistent effects
- Revenue-at-risk ranking surfaces a clear priority segment: high churn
  probability combined with R600-800 monthly spend

## Recommendation
Retention should prioritize customers by revenue-at-risk score rather than
raw churn probability. Contract type and tenure being the dominant drivers
suggests a proactive contract-renewal incentive for month-to-month customers
approaching the 6-10 month mark, addressing the problem before it reaches the
retention call stage at all.

## Note on data
Base dataset is the Telco Customer Churn dataset, rescaled to realistic South
African postpaid mobile ARPU bands to reflect the local market context this
analysis is framed around.

## How to run
Open in Google Colab, run top to bottom. Kaggle authentication uses Colab
Secrets (see notebook for setup). See requirements.txt for package versions.
