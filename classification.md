# Table of Contents

- [Classification](#classification)
  - [1. What is a Linear Classifier?](#1-what-is-a-linear-classifier)
  - [2. Decision Boundaries in Linear Classifiers](#2-decision-boundaries-in-linear-classifiers)
  - [3. Logistic Regression: Moving Beyond Just Labels](#3-logistic-regression-moving-beyond-just-labels)
  - [4. Quick Review of Probability Concepts](#4-quick-review-of-probability-concepts)
  - [5. Logistic Regression & the Sigmoid Function](#5-logistic-regression--the-sigmoid-function)
  - [6. Learning Logistic Regression from Data](#6-learning-logistic-regression-from-data)
  - [7. Categorical Features & One-Hot Encoding](#7-categorical-features--one-hot-encoding)
  - [8. Multi-Class Classification with One-vs-All](#8-multi-class-classification-with-one-vs-all)
  - [9. Quick Recap of Logistic Regression](#9-quick-recap-of-logistic-regression)
  - [10. Learning the Parameters (\( w \))](#10-learning-the-parameters-w)
  - [11. The Likelihood Function](#11-the-likelihood-function)
  - [12. Why Use Log Likelihood Instead?](#12-why-use-log-likelihood-instead)
  - [13. Optimization: Gradient Ascent](#13-optimization-gradient-ascent)
  - [14. How the Gradient Works](#14-how-the-gradient-works)
  - [15. Step Size (\( \eta \)) & Learning Rate Tuning](#15-step-size--learning-rate-tuning)
  - [16. Interpretation of Updates](#16-interpretation-of-updates)
  - [17. Final Gradient Ascent Algorithm for Logistic Regression](#17-final-gradient-ascent-algorithm-for-logistic-regression)
  - [Understanding Overfitting in Classification](#understanding-overfitting-in-classification)
  - [Measuring Classification Error](#measuring-classification-error)
  - [How Overfitting Looks in Classification](#how-overfitting-looks-in-classification)
  - [Overconfidence in Overfit Models](#overconfidence-in-overfit-models)
  - [Regularization: Preventing Overfitting](#regularization-preventing-overfitting)
  - [How Regularization Improves Generalization](#how-regularization-improves-generalization)
  - [Implementing Regularization with Gradient Ascent](#implementing-regularization-with-gradient-ascent)
  - [L1 Regularization & Sparsity](#l1-regularization--sparsity)
  - [How to Choose the Right Regularization Parameter (\( \lambda \))](#how-to-choose-the-right-regularization-parameter-)
  - [The Reality of Missing Data](#the-reality-of-missing-data)
  - [Common Approaches to Handling Missing Data](#common-approaches-to-handling-missing-data)
  - [Approach 1: Skipping Missing Data (Purification)](#approach-1-skipping-missing-data-purification)
  - [Approach 2: Imputation (Filling in Missing Values)](#approach-2-imputation-filling-in-missing-values)
  - [Approach 3: Modifying Decision Trees to Handle Missing Data](#approach-3-modifying-decision-trees-to-handle-missing-data)
  - [Making Decision Trees Robust to Missing Data](#making-decision-trees-robust-to-missing-data)
  - [Example: Handling a Loan Application with Missing Income](#example-handling-a-loan-application-with-missing-income)
  - [Optimizing Decision Trees for Missing Data](#optimizing-decision-trees-for-missing-data)
  - [What is Boosting?](#what-is-boosting)
  - [The Idea Behind Boosting](#the-idea-behind-boosting)
  - [How Boosting Works](#how-boosting-works)
  - [Understanding the AdaBoost Algorithm](#understanding-the-adaboost-algorithm)
  - [How AdaBoost Improves Accuracy](#how-adaboost-improves-accuracy)
  - [Why Does AdaBoost Work?](#why-does-adaboost-work)
  - [Overfitting in Boosting](#overfitting-in-boosting)
  - [Comparison: AdaBoost vs. Random Forest](#comparison-adaboost-vs-random-forest)
  - [Boosting in the Real World](#boosting-in-the-real-world)
  - [Why Accuracy is Not Enough?](#why-accuracy-is-not-enough)
  - [The Restaurant Review Example](#the-restaurant-review-example)
  - [Understanding Precision and Recall](#understanding-precision-and-recall)
  - [Precision vs. Recall Trade-off](#precision-vs-recall-trade-off)
  - [False Positives vs. False Negatives](#false-positives-vs-false-negatives)
  - [Optimizing the Trade-off: Adjusting the Threshold](#optimizing-the-trade-off-adjusting-the-threshold)
  - [Precision-Recall Curve](#precision-recall-curve)
  - [Precision@K: A Practical Metric](#precisionk-a-practical-metric)
  - [The Challenge of Large Datasets](#the-challenge-of-large-datasets)
  - [Why Traditional Gradient Descent Fails on Big Data](#why-traditional-gradient-descent-fails-on-big-data)
  - [Stochastic Gradient Descent (SGD) – A Game Changer](#stochastic-gradient-descent-sgd--a-game-changer)
  - [Comparing Gradient Descent vs. Stochastic Gradient Descent](#comparing-gradient-descent-vs-stochastic-gradient-descent)
  - [The Role of Online Learning](#the-role-of-online-learning)
  - [Practical Challenges with SGD & Online Learning](#practical-challenges-with-sgd--online-learning)
  - [Distributed & Parallel Machine Learning](#distributed--parallel-machine-learning)


# Classification

Classification is one of the most widely used and fundamental areas of Machine Learning. It focuses on building models that assign discrete labels to inputs.

A classifier maps input features (X) to an output class (Y).

**Examples:**
- Spam filters classify emails as spam or not spam.
- Multi-class classification: Categorizing content (e.g., an ad system determining if a webpage is about finance, education, or technology).
- Image classification: Predicting the category of an image (e.g., dog breeds in ImageNet).
- Medical applications: Personalized medicine can use classification models to predict the best treatment based on DNA and lifestyle.
- Mind-reading models: fMRI-based classifiers can predict what a person is thinking by analyzing brain scans.

**Why is Classification Important?**
- Used everywhere: spam detection, web search ranking, medical diagnosis, and recommendation systems.
- Core techniques apply to almost all supervised learning problems.

## 1. What is a Linear Classifier?

A linear classifier predicts whether an input belongs to one class (+1) or another (-1) by computing a weighted sum of input features.

**Example: Sentiment Analysis**
- A classifier assigns weights to words in a review.
- Positive words (e.g., "awesome") have positive weights.
- Negative words (e.g., "awful") have negative weights.
- If the overall score is > 0, it’s classified as positive; otherwise, it's negative.

## 2. Decision Boundaries in Linear Classifiers

The decision boundary is a hyperplane (a line in 2D, a plane in 3D, and so on) that separates classes.

**Example:**
- "Awesome" has a weight of +1.0, "Awful" has a weight of -1.5.
- If a sentence contains more "awesomes" than "awfuls," it gets classified as positive.
- The boundary equation:
  \[
  1.0 \times (\# \text{awesomes}) - 1.5 \times (\# \text{awfuls}) = 0
  \]
- Everything below the line is classified as positive.
- Everything above the line is classified as negative.

## 3. Logistic Regression: Moving Beyond Just Labels

Logistic Regression extends linear classifiers by predicting probabilities instead of hard labels. The output is not just "positive" or "negative" but a confidence score.

**Example:**
- “The sushi & everything was awesome!” → 99% probability of positive
- “The sushi was good, the service was okay.” → 55% probability of positive

This is useful when some classifications are uncertain.

## 4. Quick Review of Probability Concepts

- Probability is always between 0 and 1.
- Sum of all probabilities = 1.
- Conditional probability: The probability of an event given some condition.

**Example:**
\[
P(\text{review is positive} \mid \text{contains "awesome"}) = 0.9
\]
This means that, given a review contains "awesome," 90% of such reviews are positive.

## 5. Logistic Regression & the Sigmoid Function

How do we convert a score (which ranges from -∞ to +∞) into a probability (0 to 1)? The answer: The Sigmoid Function.

**Formula:**
\[
P(y = +1 \mid x) = \frac{1}{1 + e^{-w^T x}}
\]

**Properties:**
- If score → +∞, probability → 1 (very confident positive).
- If score → -∞, probability → 0 (very confident negative).
- If score = 0, probability = 0.5 (uncertain).

Sigmoid acts as a "squashing function," keeping outputs between 0 and 1.

## 6. Learning Logistic Regression from Data

**Process:**
- Training Data → Feature Extraction → Learn Parameters \( w \) → Predict Sentiment

**Likelihood Function:**
- Measures how good a set of parameters \( w \) is for classification.
- Example: Different decision boundaries will give different likelihood scores.

## 7. Categorical Features & One-Hot Encoding

Machine learning models handle numeric data better than categorical data (e.g., "Country" or "Gender").

**Solution:** Convert categorical variables into one-hot vectors.

**Example:**
- Instead of "Country: Brazil," we create binary features:
  - Argentina → 0
  - Brazil → 1
  - Zimbabwe → 0

**Bag-of-Words Encoding** is a common technique for text data.

**Example:**
- Sentence: "The sushi was amazing, but service was slow."
- Convert it into:
  - \( h_1 = 1 \) (number of “amazing”)
  - \( h_2 = 1 \) (number of “slow”)
  - \( h_3 = 1 \) (number of “sushi”)

## 8. Multi-Class Classification with One-vs-All

Logistic regression is binary, but what if we have more than two classes?

**One-vs-All Strategy:**
- Train one classifier per class, treating it as class vs. everything else.

**Example: Triangle, Heart, Donut**
- Classifier 1: Is it a triangle vs. not a triangle?
- Classifier 2: Is it a heart vs. not a heart?
- Classifier 3: Is it a donut vs. not a donut?

Choose the class with the highest probability.

## 9. Quick Recap of Logistic Regression

**Goal:** Learn a classifier that predicts \( P(y \mid x) \), where \( y \) is the output label (positive or negative sentiment), and \( x \) is the input (e.g., a restaurant review).

- Logistic regression assigns weights (\( w \)) to input features (e.g., words in a review) and computes a score.
- This score is passed through the sigmoid function to squash it between 0 and 1, giving a probability.

## 10. Learning the Parameters (\( w \))

**Objective:** Find the best weights (\( w \)) so that the model accurately classifies inputs.

- Training data consists of (\( x, y \)) pairs, where \( x \) is the input, and \( y \) is the true label (positive or negative).
- We want to maximize the probability of correct classifications across all training examples.

## 11. The Likelihood Function

The likelihood function quantifies how well a model fits the data.

- Higher likelihood = better model fit.
- Given a dataset of \( N \) examples:
  - For positive examples, we maximize \( P(y = +1 \mid x, w) \).
  - For negative examples, we maximize \( P(y = -1 \mid x, w) \).
- The overall likelihood function is the product of probabilities across all training examples.

**Example: Computing Likelihood**
- Suppose we have 4 reviews:
  - (2 "awesomes", 1 "awful", positive review) → Want high \( P(y = +1 \mid x) \)
  - (0 "awesomes", 2 "awfuls", negative review) → Want high \( P(y = -1 \mid x) \)

**Likelihood function:**
\[
L(w) = P(y_1 \mid x_1, w) \times P(y_2 \mid x_2, w) \times \ldots \times P(y_N \mid x_N, w)
\]
We seek \( w \) that maximizes this likelihood function.

## 12. Why Use Log Likelihood Instead?

Since likelihood is a product of many probabilities, it results in very small values. Instead of maximizing \( L(w) \), we maximize its log (log-likelihood function), which turns the product into a sum:

\[
\log L(w) = \sum_{i=1}^{N} \log P(y_i \mid x_i, w)
\]

Taking the log simplifies calculations and makes optimization easier.

## 13. Optimization: Gradient Ascent

Gradient ascent is used to find the optimal weights \( w \) that maximize the log-likelihood function.

**Key idea:** Start with random weights and iteratively adjust them in the direction of increasing likelihood.

**Gradient Ascent Algorithm:**
1. Initialize \( w \) randomly.
2. Compute the gradient (how much each weight should change).
3. Update each weight using:
   \[
   w_j = w_j + \eta \cdot \frac{\partial \log L}{\partial w_j}
   \]
   - \( \eta \) (step size) controls how big the update is.
4. Stop when updates become small (converged).

## 14. How the Gradient Works

The gradient measures the direction and magnitude of change needed for each weight.

**Formula for updating \( w_j \):**
\[
\frac{\partial \log L}{\partial w_j} = \sum_{i=1}^{N} (y_i - P(y_i = +1 \mid x_i, w)) \cdot x_{ij}
\]

**Interpretation:**
- If a positive review is misclassified as negative, increase weights for positive words.
- If a negative review is misclassified as positive, decrease weights for negative words.
- If a review is classified correctly, don’t change much.

## 15. Step Size (\( \eta \)) & Learning Rate Tuning

Choosing the right step size (\( \eta \)) is crucial.

- If \( \eta \) is too small: The model learns slowly, taking too long to converge.
- If \( \eta \) is too large: The model oscillates or diverges, failing to find the optimal weights.

**How to Find the Best Step Size?**
- Learning curves: Track log-likelihood over iterations.
  - Too small \( \eta \) → Converges slowly.
  - Too large \( \eta \) → Wild oscillations.
  - Optimal \( \eta \) → Smooth increase in log-likelihood, reaching a maximum efficiently.

## 16. Interpretation of Updates

- If a training example is classified correctly, little to no change in weights.
- If an example is misclassified:
  - If it should be positive, increase its word weights.
  - If it should be negative, decrease its word weights.

**Example Calculation:**
- Suppose a review has 2 “awesomes”, 1 “awful”, and is positive.
- If the current model predicts 0.5 probability, the update is:
  \[
  w_j = w_j + \eta \cdot (1 - 0.5) \cdot 2
  \]
  Increase \( w_j \) since the model was uncertain about a positive review.

## 17. Final Gradient Ascent Algorithm for Logistic Regression

1. Initialize weights \( w \) randomly.
2. Repeat until convergence:
   - For each weight \( w_j \), update using:
     \[
     w_j = w_j + \eta \sum_{i=1}^{N} (y_i - P(y_i = +1 \mid x_i, w)) \cdot x_{ij}
     \]
3. Stop when updates are very small.

**Note:** The algorithm iteratively adjusts weights to maximize the log-likelihood function, converging to the best weights for the model.

## Understanding Overfitting in Classification

Overfitting happens when a model performs well on training data but fails to generalize to new data. In classification, overfitting can cause:
- Overly complex decision boundaries.
- Extremely high confidence in wrong predictions.

**Example:** A classifier that perfectly fits training data might memorize noise rather than learn general patterns.

## Measuring Classification Error

We evaluate classifiers using classification error:
\[ \text{Error} = \frac{\text{Number of misclassified examples}}{\text{Total number of examples}} \]

Accuracy = 1 - Error. Classifiers should be evaluated on a separate validation set to detect overfitting.

## How Overfitting Looks in Classification

Overfitting can happen in logistic regression when decision boundaries become too complex.

**Example:**
- A simple linear classifier correctly separates most data.
- A quadratic classifier captures more nuances, improving accuracy.
- A high-degree polynomial (e.g., degree 20) creates wildly complex decision boundaries.

**Problem:** Complex boundaries don’t generalize and may misclassify new data.

**Key Signs of Overfitting:**
- Decision boundaries become highly irregular.
- Large coefficient values (weights become extreme).
- Extreme confidence (probabilities near 0 or 1) for uncertain cases.

## Overconfidence in Overfit Models

Logistic regression models output probabilities, but overfit models push probabilities toward 0 or 1.

**Example:** A review with 2 "awesomes" and 1 "awful" should have a reasonable probability (e.g., 73%) of being positive. If coefficients become too large, the model wrongly becomes 99.7% sure it's positive.

**Problem:** The model loses uncertainty and becomes overconfident in wrong predictions.

## Regularization: Preventing Overfitting

Solution: Regularization penalizes large weights, making the model simpler and more generalizable.

**Two types of regularization:**
- **L2 Regularization (Ridge Regression in Regression):** Penalizes large weights using sum of squared coefficients:
  \[ \lambda \sum w_j^2 \]
  Helps reduce overfitting while maintaining smooth decision boundaries.
  **Effects:**
  - Keeps coefficients small but nonzero.
  - Prevents extreme decision boundaries.
  - Balances training fit vs. generalization.

- **L1 Regularization (Lasso in Regression):** Penalizes large weights using sum of absolute values:
  \[ \lambda \sum |w_j| \]
  Encourages sparsity → many weights become exactly 0.
  Useful when working with high-dimensional data (e.g., spam detection with thousands of features).
  **Effects:**
  - Fewer active features (improves interpretability & efficiency).
  - Sparse solutions → only important features remain.

## How Regularization Improves Generalization

**Example:** Applying L2 Regularization on a Degree-20 Model
- Without regularization → crazy decision boundary, large coefficients (3000+).
- With L2 regularization → smooth, well-behaved boundary.

**Effect on Probabilities:**
- Without regularization → Overconfident probabilities (near 0 or 1).
- With regularization → Reasonable confidence levels (proper uncertainty maintained).

## Implementing Regularization with Gradient Ascent

Gradient Ascent for Regularized Logistic Regression:
The update rule adds a penalty term:
\[ w_j = w_j + \eta \left( \sum_{i=1}^{N} (y_i - P(y_i = +1 \mid x_i, w)) x_{ij} - 2\lambda w_j \right) \]

**Effect:**
- Reduces large coefficients over time.
- Helps maintain smooth, generalizable decision boundaries.

**Implementation Change:** Only one small change! Add \(-2\lambda w_j\) to your existing gradient ascent code.

## L1 Regularization & Sparsity

L1 Regularization shrinks some coefficients to exactly zero.

**Why is this useful?**
- Reduces computation (faster predictions).
- Improves interpretability (removes unnecessary features).

**Example: Word Importance in Sentiment Analysis**
- L2 Regularization: Words like “great”, “bad”, and “disappointed” get small weights.
- L1 Regularization: Some words get completely removed (zero weight), leaving only the most important.

## How to Choose the Right Regularization Parameter (λ)?

- λ too small → Overfitting still happens.
- λ too large → Model oversimplifies (underfitting).

**Best approach:** Use cross-validation or a validation set to tune λ.

## The Reality of Missing Data

In previous modules, we assumed that every data point had all feature values.
In reality, datasets are often incomplete:
- Loan applications may be missing income or credit history.
- Medical records may lack certain test results.
- Customer data may have unknown age, address, etc.

Missing data can impact ML models at:
- Training time → We can’t properly train if features are missing.
- Prediction time → The model doesn’t know what to do with missing inputs.

## Common Approaches to Handling Missing Data

We discuss three main strategies:

| Approach | Pros | Cons |
|----------|------|------|
| Skipping Missing Data | Simple, easy to implement | Reduces dataset size, risks removing valuable information |
| Imputation (Filling Missing Values) | Preserves data | Introduces bias, makes incorrect assumptions |
| Modifying the Model to Handle Missing Data | More accurate, adapts to missing values | Requires modifying algorithms |

## Approach 1: Skipping Missing Data (Purification)

The simplest solution: Remove rows or features with missing values.

**Two options:**
1. Skip rows with missing values (Reduces dataset size).
2. Skip entire features if too many values are missing.

**Example**
| Credit | Term | Income | Risky Loan? |
|--------|------|--------|-------------|
| Poor | 3Y | High | Yes |
| Fair | ? | High | No |
| Excellent | 5Y | ? | No |

- If only a few rows have missing values, dropping them is fine.
- If many values are missing, removing them may destroy useful data.

**Problems with Skipping Data**
- ❌ If many rows are removed, we lose valuable information.
- ❌ If many features are dropped, we may miss important patterns.
- ❌ Doesn’t solve the issue at prediction time (what if a missing value appears in new data?).

✅ Good when missing values are rare, but not a scalable solution.

## Approach 2: Imputation (Filling in Missing Values)

Instead of removing missing data, we fill it in using estimated values.

**Example:** If 70% of loans are 3-year loans, replace missing “Term” values with 3 years.

**Common Imputation Strategies:**
- For Categorical Features (e.g., Credit Score: Excellent/Fair/Poor)
  - Replace missing values with the most common category.
- For Numerical Features (e.g., Income)
  - Replace missing values with the mean/median of the observed values.

**Problems with Simple Imputation**
- ❌ Can introduce bias (e.g., assuming everyone missing "age" is 40).
- ❌ Can misrepresent reality (e.g., assuming all missing loans are 3-year loans).
- ❌ May not be correct at prediction time.

✅ Better than skipping data, but introduces systematic errors.

## Approach 3: Modifying Decision Trees to Handle Missing Data

Instead of skipping or guessing, modify decision trees to handle missing values natively.

**How Does This Work?**
- Allow the model to learn how to deal with missing values.
- Modify decision trees to assign missing values to a specific branch.
- **Example:** If Income is missing, send it down the "Low Income" branch.
- This optimizes tree splits to minimize classification error.

## Making Decision Trees Robust to Missing Data

Normally, decision trees split on features (e.g., Credit Score).
But if Credit Score is missing, what should we do?

**Solution:** Assign missing values to the best branch based on past# Classification

Classification is one of the most widely used and fundamental areas of Machine Learning. It focuses on building models that assign discrete labels to inputs.

A classifier maps input features (X) to an output class (Y).

**Examples:**
- Spam filters classify emails as spam or not spam.
- Multi-class classification: Categorizing content (e.g., an ad system determining if a webpage is about finance, education, or technology).
- Image classification: Predicting the category of an image (e.g., dog breeds in ImageNet).
- Medical applications: Personalized medicine can use classification models to predict the best treatment based on DNA and lifestyle.
- Mind-reading models: fMRI-based classifiers can predict what a person is thinking by analyzing brain scans.

**Why is Classification Important?**
- Used everywhere: spam detection, web search ranking, medical diagnosis, and recommendation systems.
- Core techniques apply to almost all supervised learning problems.

## 1. What is a Linear Classifier?

A linear classifier predicts whether an input belongs to one class (+1) or another (-1) by computing a weighted sum of input features.

**Example: Sentiment Analysis**
- A classifier assigns weights to words in a review.
- Positive words (e.g., "awesome") have positive weights.
- Negative words (e.g., "awful") have negative weights.
- If the overall score is > 0, it’s classified as positive; otherwise, it's negative.

## 2. Decision Boundaries in Linear Classifiers

The decision boundary is a hyperplane (a line in 2D, a plane in 3D, and so on) that separates classes.

**Example:**
- "Awesome" has a weight of +1.0, "Awful" has a weight of -1.5.
- If a sentence contains more "awesomes" than "awfuls," it gets classified as positive.
- The boundary equation:
  \[
  1.0 \times (\# \text{awesomes}) - 1.5 \times (\# \text{awfuls}) = 0
  \]
- Everything below the line is classified as positive.
- Everything above the line is classified as negative.

## 3. Logistic Regression: Moving Beyond Just Labels

Logistic Regression extends linear classifiers by predicting probabilities instead of hard labels. The output is not just "positive" or "negative" but a confidence score.

**Example:**
- “The sushi & everything was awesome!” → 99% probability of positive
- “The sushi was good, the service was okay.” → 55% probability of positive

This is useful when some classifications are uncertain.

## 4. Quick Review of Probability Concepts

- Probability is always between 0 and 1.
- Sum of all probabilities = 1.
- Conditional probability: The probability of an event given some condition.

**Example:**
\[
P(\text{review is positive} \mid \text{contains "awesome"}) = 0.9
\]
This means that, given a review contains "awesome," 90% of such reviews are positive.

## 5. Logistic Regression & the Sigmoid Function

How do we convert a score (which ranges from -∞ to +∞) into a probability (0 to 1)? The answer: The Sigmoid Function.

**Formula:**
\[
P(y = +1 \mid x) = \frac{1}{1 + e^{-w^T x}}
\]

**Properties:**
- If score → +∞, probability → 1 (very confident positive).
- If score → -∞, probability → 0 (very confident negative).
- If score = 0, probability = 0.5 (uncertain).

Sigmoid acts as a "squashing function," keeping outputs between 0 and 1.

## 6. Learning Logistic Regression from Data

**Process:**
- Training Data → Feature Extraction → Learn Parameters \( w \) → Predict Sentiment

**Likelihood Function:**
- Measures how good a set of parameters \( w \) is for classification.
- Example: Different decision boundaries will give different likelihood scores.

## 7. Categorical Features & One-Hot Encoding

Machine learning models handle numeric data better than categorical data (e.g., "Country" or "Gender").

**Solution:** Convert categorical variables into one-hot vectors.

**Example:**
- Instead of "Country: Brazil," we create binary features:
  - Argentina → 0
  - Brazil → 1
  - Zimbabwe → 0

**Bag-of-Words Encoding** is a common technique for text data.

**Example:**
- Sentence: "The sushi was amazing, but service was slow."
- Convert it into:
  - \( h_1 = 1 \) (number of “amazing”)
  - \( h_2 = 1 \) (number of “slow”)
  - \( h_3 = 1 \) (number of “sushi”)

## 8. Multi-Class Classification with One-vs-All

Logistic regression is binary, but what if we have more than two classes?

**One-vs-All Strategy:**
- Train one classifier per class, treating it as class vs. everything else.

**Example: Triangle, Heart, Donut**
- Classifier 1: Is it a triangle vs. not a triangle?
- Classifier 2: Is it a heart vs. not a heart?
- Classifier 3: Is it a donut vs. not a donut?

Choose the class with the highest probability.

## 9. Quick Recap of Logistic Regression

**Goal:** Learn a classifier that predicts \( P(y \mid x) \), where \( y \) is the output label (positive or negative sentiment), and \( x \) is the input (e.g., a restaurant review).

- Logistic regression assigns weights (\( w \)) to input features (e.g., words in a review) and computes a score.
- This score is passed through the sigmoid function to squash it between 0 and 1, giving a probability.

## 10. Learning the Parameters (\( w \))

**Objective:** Find the best weights (\( w \)) so that the model accurately classifies inputs.

- Training data consists of (\( x, y \)) pairs, where \( x \) is the input, and \( y \) is the true label (positive or negative).
- We want to maximize the probability of correct classifications across all training examples.

## 11. The Likelihood Function

The likelihood function quantifies how well a model fits the data.

- Higher likelihood = better model fit.
- Given a dataset of \( N \) examples:
  - For positive examples, we maximize \( P(y = +1 \mid x, w) \).
  - For negative examples, we maximize \( P(y = -1 \mid x, w) \).
- The overall likelihood function is the product of probabilities across all training examples.

**Example: Computing Likelihood**
- Suppose we have 4 reviews:
  - (2 "awesomes", 1 "awful", positive review) → Want high \( P(y = +1 \mid x) \)
  - (0 "awesomes", 2 "awfuls", negative review) → Want high \( P(y = -1 \mid x) \)

**Likelihood function:**
\[
L(w) = P(y_1 \mid x_1, w) \times P(y_2 \mid x_2, w) \times \ldots \times P(y_N \mid x_N, w)
\]
We seek \( w \) that maximizes this likelihood function.

## 12. Why Use Log Likelihood Instead?

Since likelihood is a product of many probabilities, it results in very small values. Instead of maximizing \( L(w) \), we maximize its log (log-likelihood function), which turns the product into a sum:

\[
\log L(w) = \sum_{i=1}^{N} \log P(y_i \mid x_i, w)
\]

Taking the log simplifies calculations and makes optimization easier.

## 13. Optimization: Gradient Ascent

Gradient ascent is used to find the optimal weights \( w \) that maximize the log-likelihood function.

**Key idea:** Start with random weights and iteratively adjust them in the direction of increasing likelihood.

**Gradient Ascent Algorithm:**
1. Initialize \( w \) randomly.
2. Compute the gradient (how much each weight should change).
3. Update each weight using:
   \[
   w_j = w_j + \eta \cdot \frac{\partial \log L}{\partial w_j}
   \]
   - \( \eta \) (step size) controls how big the update is.
4. Stop when updates become small (converged).

## 14. How the Gradient Works

The gradient measures the direction and magnitude of change needed for each weight.

**Formula for updating \( w_j \):**
\[
\frac{\partial \log L}{\partial w_j} = \sum_{i=1}^{N} (y_i - P(y_i = +1 \mid x_i, w)) \cdot x_{ij}
\]

**Interpretation:**
- If a positive review is misclassified as negative, increase weights for positive words.
- If a negative review is misclassified as positive, decrease weights for negative words.
- If a review is classified correctly, don’t change much.

## 15. Step Size (\( \eta \)) & Learning Rate Tuning

Choosing the right step size (\( \eta \)) is crucial.

- If \( \eta \) is too small: The model learns slowly, taking too long to converge.
- If \( \eta \) is too large: The model oscillates or diverges, failing to find the optimal weights.

**How to Find the Best Step Size?**
- Learning curves: Track log-likelihood over iterations.
  - Too small \( \eta \) → Converges slowly.
  - Too large \( \eta \) → Wild oscillations.
  - Optimal \( \eta \) → Smooth increase in log-likelihood, reaching a maximum efficiently.

## 16. Interpretation of Updates

- If a training example is classified correctly, little to no change in weights.
- If an example is misclassified:
  - If it should be positive, increase its word weights.
  - If it should be negative, decrease its word weights.

**Example Calculation:**
- Suppose a review has 2 “awesomes”, 1 “awful”, and is positive.
- If the current model predicts 0.5 probability, the update is:
  \[
  w_j = w_j + \eta \cdot (1 - 0.5) \cdot 2
  \]
  Increase \( w_j \) since the model was uncertain about a positive review.

## 17. Final Gradient Ascent Algorithm for Logistic Regression

1. Initialize weights \( w \) randomly.
2. Repeat until convergence:
   - For each weight \( w_j \), update using:
     \[
     w_j = w_j + \eta \sum_{i=1}^{N} (y_i - P(y_i = +1 \mid x_i, w)) \cdot x_{ij}
     \]
3. Stop when updates are very small.

**Note:** The algorithm iteratively adjusts weights to maximize the log-likelihood function, converging to the best weights for the model.

## Understanding Overfitting in Classification

Overfitting happens when a model performs well on training data but fails to generalize to new data. In classification, overfitting can cause:
- Overly complex decision boundaries.
- Extremely high confidence in wrong predictions.

**Example:** A classifier that perfectly fits training data might memorize noise rather than learn general patterns.

## Measuring Classification Error

We evaluate classifiers using classification error:
\[ \text{Error} = \frac{\text{Number of misclassified examples}}{\text{Total number of examples}} \]

Accuracy = 1 - Error. Classifiers should be evaluated on a separate validation set to detect overfitting.

## How Overfitting Looks in Classification

Overfitting can happen in logistic regression when decision boundaries become too complex.

**Example:**
- A simple linear classifier correctly separates most data.
- A quadratic classifier captures more nuances, improving accuracy.
- A high-degree polynomial (e.g., degree 20) creates wildly complex decision boundaries.

**Problem:** Complex boundaries don’t generalize and may misclassify new data.

**Key Signs of Overfitting:**
- Decision boundaries become highly irregular.
- Large coefficient values (weights become extreme).
- Extreme confidence (probabilities near 0 or 1) for uncertain cases.

## Overconfidence in Overfit Models

Logistic regression models output probabilities, but overfit models push probabilities toward 0 or 1.

**Example:** A review with 2 "awesomes" and 1 "awful" should have a reasonable probability (e.g., 73%) of being positive. If coefficients become too large, the model wrongly becomes 99.7% sure it's positive.

**Problem:** The model loses uncertainty and becomes overconfident in wrong predictions.

## Regularization: Preventing Overfitting

Solution: Regularization penalizes large weights, making the model simpler and more generalizable.

**Two types of regularization:**
- **L2 Regularization (Ridge Regression in Regression):** Penalizes large weights using sum of squared coefficients:
  \[ \lambda \sum w_j^2 \]
  Helps reduce overfitting while maintaining smooth decision boundaries.
  **Effects:**
  - Keeps coefficients small but nonzero.
  - Prevents extreme decision boundaries.
  - Balances training fit vs. generalization.

- **L1 Regularization (Lasso in Regression):** Penalizes large weights using sum of absolute values:
  \[ \lambda \sum |w_j| \]
  Encourages sparsity → many weights become exactly 0.
  Useful when working with high-dimensional data (e.g., spam detection with thousands of features).
  **Effects:**
  - Fewer active features (improves interpretability & efficiency).
  - Sparse solutions → only important features remain.

## How Regularization Improves Generalization

**Example:** Applying L2 Regularization on a Degree-20 Model
- Without regularization → crazy decision boundary, large coefficients (3000+).
- With L2 regularization → smooth, well-behaved boundary.

**Effect on Probabilities:**
- Without regularization → Overconfident probabilities (near 0 or 1).
- With regularization → Reasonable confidence levels (proper uncertainty maintained).

## Implementing Regularization with Gradient Ascent

Gradient Ascent for Regularized Logistic Regression:
The update rule adds a penalty term:
\[ w_j = w_j + \eta \left( \sum_{i=1}^{N} (y_i - P(y_i = +1 \mid x_i, w)) x_{ij} - 2\lambda w_j \right) \]

**Effect:**
- Reduces large coefficients over time.
- Helps maintain smooth, generalizable decision boundaries.

**Implementation Change:** Only one small change! Add \(-2\lambda w_j\) to your existing gradient ascent code.

## L1 Regularization & Sparsity

L1 Regularization shrinks some coefficients to exactly zero.

**Why is this useful?**
- Reduces computation (faster predictions).
- Improves interpretability (removes unnecessary features).

**Example: Word Importance in Sentiment Analysis**
- L2 Regularization: Words like “great”, “bad”, and “disappointed” get small weights.
- L1 Regularization: Some words get completely removed (zero weight), leaving only the most important.

## How to Choose the Right Regularization Parameter (λ)?

- λ too small → Overfitting still happens.
- λ too large → Model oversimplifies (underfitting).

**Best approach:** Use cross-validation or a validation set to tune λ.

## The Reality of Missing Data

In previous modules, we assumed that every data point had all feature values.
In reality, datasets are often incomplete:
- Loan applications may be missing income or credit history.
- Medical records may lack certain test results.
- Customer data may have unknown age, address, etc.

Missing data can impact ML models at:
- Training time → We can’t properly train if features are missing.
- Prediction time → The model doesn’t know what to do with missing inputs.

## Common Approaches to Handling Missing Data

We discuss three main strategies:

| Approach | Pros | Cons |
|----------|------|------|
| Skipping Missing Data | Simple, easy to implement | Reduces dataset size, risks removing valuable information |
| Imputation (Filling Missing Values) | Preserves data | Introduces bias, makes incorrect assumptions |
| Modifying the Model to Handle Missing Data | More accurate, adapts to missing values | Requires modifying algorithms |

## Approach 1: Skipping Missing Data (Purification)

The simplest solution: Remove rows or features with missing values.

**Two options:**
1. Skip rows with missing values (Reduces dataset size).
2. Skip entire features if too many values are missing.

**Example**
| Credit | Term | Income | Risky Loan? |
|--------|------|--------|-------------|
| Poor | 3Y | High | Yes |
| Fair | ? | High | No |
| Excellent | 5Y | ? | No |

- If only a few rows have missing values, dropping them is fine.
- If many values are missing, removing them may destroy useful data.

**Problems with Skipping Data**
- ❌ If many rows are removed, we lose valuable information.
- ❌ If many features are dropped, we may miss important patterns.
- ❌ Doesn’t solve the issue at prediction time (what if a missing value appears in new data?).

✅ Good when missing values are rare, but not a scalable solution.

## Approach 2: Imputation (Filling in Missing Values)

Instead of removing missing data, we fill it in using estimated values.

**Example:** If 70% of loans are 3-year loans, replace missing “Term” values with 3 years.

**Common Imputation Strategies:**
- For Categorical Features (e.g., Credit Score: Excellent/Fair/Poor)
  - Replace missing values with the most common category.
- For Numerical Features (e.g., Income)
  - Replace missing values with the mean/median of the observed values.

**Problems with Simple Imputation**
- ❌ Can introduce bias (e.g., assuming everyone missing "age" is 40).
- ❌ Can misrepresent reality (e.g., assuming all missing loans are 3-year loans).
- ❌ May not be correct at prediction time.

✅ Better than skipping data, but introduces systematic errors.

## Approach 3: Modifying Decision Trees to Handle Missing Data

Instead of skipping or guessing, modify decision trees to handle missing values natively.

**How Does This Work?**
- Allow the model to learn how to deal with missing values.
- Modify decision trees to assign missing values to a specific branch.
- **Example:** If Income is missing, send it down the "Low Income" branch.
- This optimizes tree splits to minimize classification error.

## Making Decision Trees Robust to Missing Data

Normally, decision trees split on features (e.g., Credit Score).
But if Credit Score is missing, what should we do?

**Solution:** Assign missing values to the best branch based on past


## Example: Handling a Loan Application with Missing Income

| Credit | Term | Income | Risky Loan? |
|--------|------|--------|-------------|
| Poor   | 3Y   | ?      | Yes         |

### Regular Decision Tree

- Split on Credit Score → Poor.
- Split on Income → Missing Value → ❌ Can’t Proceed.

### Modified Decision Tree

- Split on Credit Score → Poor.
- Income is Missing → Send to “Low Income” branch (or another branch based on error minimization).
- Prediction is made despite missing data.

## Optimizing Decision Trees for Missing Data

Every decision node in the tree decides where to send missing values. The algorithm learns from data where missing values should go.

### Key Idea

Choose the branch that minimizes classification error.

### Example: Choosing the Best Split for Missing Data

1. Try placing missing values in Branch A.
2. Try placing missing values in Branch B.
3. Choose the branch that results in lower classification error.

### Advantages of This Approach

- Works both at training and prediction time.
- Doesn’t remove data (no lost information).
- More accurate than imputation.
- Automatically handles missing data in a meaningful way.

## What is Boosting?

Boosting is a meta-learning technique that enhances the performance of weak classifiers. It iteratively trains weak models, giving more weight to the misclassified examples in each round.

**Common weak classifiers used in boosting:**
- Decision Stumps (shallow decision trees)
- Logistic Regression

### Why Use Boosting?

- ✔ Reduces bias (improves weak classifiers)
- ✔ Lowers variance (reduces overfitting)
- ✔ Outperforms individual models
- ✔ Wins machine learning competitions (Kaggle, KDD Cup, etc.)

## The Idea Behind Boosting

Instead of training a single complex model, boosting trains multiple simple models and combines their outputs.

**Example: Loan Classification**
- Classifier 1: Splits on income
- Classifier 2: Splits on credit history
- Classifier 3: Splits on loan term
- Final Prediction: Weighted combination of all classifiers

### Ensemble Learning: The Core of Boosting

- Instead of relying on a single decision tree, boosting combines multiple models.
- Each model corrects errors made by the previous one.
- Final decision = weighted vote of all classifiers.

## How Boosting Works

1. Train a weak classifier (e.g., decision stump) on the dataset.
2. Check which examples it misclassified.
3. Increase the importance (weight) of misclassified examples.
4. Train a new classifier with updated weights.
5. Repeat this process for T iterations.
6. Combine all classifiers into a strong model.

### Boosting vs. Traditional Models

| Method              | Strengths                                      | Weaknesses                        |
|---------------------|------------------------------------------------|-----------------------------------|
| Logistic Regression | Simple, interpretable, works well for linear data | Cannot model complex patterns     |
| Decision Trees      | Handles non-linear data, interpretable         | Prone to overfitting              |
| Boosting            | High accuracy, reduces bias & variance         | Requires tuning, sensitive to noise |

## Understanding the AdaBoost Algorithm

AdaBoost (Adaptive Boosting) was one of the first practical boosting algorithms.

- Developed by Freund & Schapire in 1999.
- Works by iteratively improving weak classifiers.
- Uses a weighted voting system to combine weak models.

### AdaBoost Algorithm

1. Initialize equal weights for all training examples.
2. Train a weak classifier \( f_t(x) \) on the weighted dataset.
3. Compute the weighted error:
   \[
   \text{error} = \sum (\text{weight of misclassified points})
   \]
4. Compute the classifier’s importance:
   \[
   w_t = \frac{1}{2} \ln \left( \frac{1 - \text{error}}{\text{error}} \right)
   \]
5. Update example weights:
   - Increase weight of misclassified points.
   - Decrease weight of correctly classified points.
6. Repeat for T iterations.
7. Final prediction is based on weighted sum of classifiers.

## How AdaBoost Improves Accuracy

**Example: Face Detection**

- Classifier 1 detects edges.
- Classifier 2 detects eyes.
- Classifier 3 detects nose & mouth.
- Final ensemble model detects faces with high accuracy.

### Key Insights

- Early classifiers handle simple cases.
- Later classifiers correct mistakes.
- Final model is highly accurate.

## Why Does AdaBoost Work?

- Focuses on hard-to-classify examples.
- Combines multiple weak models into a strong one.
- Automatically assigns higher weights to better classifiers.

### AdaBoost Theorem

- Guarantees that training error decreases to zero as iterations increase.
- However, if boosting runs too long, it may overfit.

## Overfitting in Boosting

Unlike decision trees, boosting is resistant to overfitting. However, if too many weak classifiers are added, test error may increase.

### How to Prevent Overfitting?

- ✔ Use cross-validation to choose the optimal number of iterations (T).
- ✔ Regularization (limit complexity of weak learners).
- ✔ Early stopping (stop training when validation error increases).

## Comparison: AdaBoost vs. Random Forest

| Method           | How It Works                                             | Strengths                          | Weaknesses                        |
|------------------|----------------------------------------------------------|-----------------------------------|-----------------------------------|
| AdaBoost         | Sequentially trains weak models, adjusts weights based on mistakes | High accuracy, focuses on hard examples | Sensitive to noise                |
| Random Forest    | Trains multiple trees in parallel, each on a random subset of data | Robust to noise, easy to parallelize | May require more trees than boosting |
| Gradient Boosting| Like AdaBoost but uses gradient descent to optimize      | Even better performance than AdaBoost | More computationally expensive     |

## Boosting in the Real World

Boosting is widely used in:
- ✔ Computer Vision – Face detection, object recognition.
- ✔ Search Engines – Ranking web pages.
- ✔ Fraud Detection – Identifying credit card fraud.
- ✔ Recommender Systems – Netflix movie recommendations.
- ✔ Finance & Healthcare – Loan approvals, disease prediction.

### Kaggle Competitions

Boosting wins over 50% of machine learning competitions.

**Popular libraries:**
- XGBoost (Extreme Gradient Boosting) – Fast & powerful.
- LightGBM – Scalable boosting for large datasets.
- CatBoost – Optimized for categorical data.



# Why Accuracy is Not Enough?

Accuracy is often misleading, especially in imbalanced datasets.

### Example: The Problem with Accuracy
Suppose 90% of restaurant reviews are negative. A classifier that always predicts "negative" achieves 90% accuracy – but it’s useless! It never identifies positive reviews, which are essential for a marketing campaign.

**Solution:** Use **Precision & Recall**, which provide a more meaningful evaluation.

---

## The Restaurant Review Example

Suppose a restaurant wants to automatically highlight positive reviews to attract customers.

- **Goal:** Extract positive sentences from user reviews and display them on the website.
- **Model:** A sentiment classifier that predicts whether a sentence is positive or negative.
- **Problem:** How do we trust this classifier?
  - If it fails, it might post a negative review on the website (very bad for business!).
  - **Accuracy alone doesn’t tell us how reliable it is.**

---

## Understanding Precision and Recall

To evaluate a classifier, we need two key metrics:

| Metric     | Definition | Why It Matters? |
|------------|----------------------------------|----------------------------------------------------------|
| **Precision** | Out of all sentences predicted as positive, how many are actually positive? | Ensures we don’t show negative reviews on the website. |
| **Recall** | Out of all actual positive sentences, how many did we correctly find? | Ensures we don’t miss good reviews. |

### Example: Precision & Recall in Action

A classifier identifies 6 sentences as "positive".
- **Reality:**
  - 4 of them are truly positive (✅).
  - 2 are actually negative (❌ - false positives).
  - Another 2 positive sentences were missed (❌ - false negatives).

#### 📌 Precision Calculation:

\[
Precision = \frac{True\ Positives}{True\ Positives + False\ Positives} = \frac{4}{6} = 0.67
\]

**Interpretation:** 67% of displayed sentences are actually positive (the rest are mistakes).

#### 📌 Recall Calculation:

\[
Recall = \frac{True\ Positives}{True\ Positives + False\ Negatives} = \frac{4}{6} = 0.67
\]

**Interpretation:** The model found 67% of all possible positive reviews (but missed some).

---

## Precision vs. Recall Trade-off

- **High Precision, Low Recall:** The model is very selective (only picks the safest positive reviews).
- **High Recall, Low Precision:** The model finds most positive reviews but also includes some mistakes.
- **Optimizing both is difficult** – you usually have to trade one for the other.

### Real-World Analogy: Spam Filter

| Scenario | Effect |
|------------|-----------------------------------------------------|
| **High Precision, Low Recall** | Blocks only obvious spam, but some spam still gets through. |
| **High Recall, Low Precision** | Blocks all spam but also blocks important emails. |

---

## False Positives vs. False Negatives

Errors in classification come in two types:

| Error Type | Definition | Example (Restaurant Reviews) | Impact |
|------------|----------------------------------|-----------------------------------------------------|-------------------------------------|
| **False Positive (FP)** | Predict positive, but it’s actually negative. | “The sushi was awful” is mistakenly posted as a positive review. | **Big problem!** Could damage business reputation. |
| **False Negative (FN)** | Predict negative, but it’s actually positive. | “The sushi was amazing” is ignored. | **Lost opportunity** to attract customers. |

For the restaurant website, **false positives are worse** (we don’t want bad reviews on the site!). Thus, **high precision is more important than high recall.**

---

## Optimizing the Trade-off: Adjusting the Threshold

Most classifiers predict a **probability score** (e.g., 0.99 → highly positive, 0.55 → uncertain).

- **Decision Threshold** (default = 0.5) determines when to classify as positive.

### How Adjusting the Threshold Changes Precision & Recall

| Threshold | Effect | Classifier Behavior |
|------------|-------------------------------|--------------------------------------------------|
| **T = 0.99 (Very High)** | High Precision, Low Recall | Only very confident predictions are labeled positive. |
| **T = 0.50 (Balanced)** | Moderate Precision & Recall | Standard decision boundary. |
| **T = 0.01 (Very Low)** | Low Precision, High Recall | Almost everything is classified as positive. |

🚀 **Application:**
- If we want **high precision** (avoid false positives), use a **higher threshold**.
- If we want **high recall** (find all positive reviews), use a **lower threshold**.

---

## Precision-Recall Curve

The **Precision-Recall Curve** shows the trade-off between precision & recall at different thresholds.

- **Better classifiers** have curves closer to the top-right corner (high precision and recall).
- We compare models using Precision-Recall curves instead of a single accuracy value.

### Example: Comparing Two Classifiers

- **Classifier A** is better overall (higher precision at every recall level).
- **Classifier B** is only better for high recall cases.

📌 **Choosing the Best Model:**
- If we care about **precision** (avoiding false positives) → Pick **Classifier A**.
- If we care about **recall** (finding all positives) → Pick **Classifier B**.

---

## Precision@K: A Practical Metric

**Precision@K** measures precision for the **top K** predictions (e.g., top 5 sentences displayed on a website).

### Example:
- We show **5 reviews**.
- **4 are correct, 1 is negative** → **Precision@5 = 4/5 = 0.8**.

### This metric is useful for:
- **Search engines** (top 10 results should be relevant).
- **Recommender systems** (top 5 recommended products should be useful).
- **Chatbot responses** (only show the best replies).

# The Challenge of Large Datasets

Machine learning models must handle massive amounts of data:

- **4.8 billion** web pages
- **500 million** tweets per day
- **YouTube:** 300 hours of video uploaded every minute

Traditional learning algorithms struggle because they require multiple passes over the entire dataset before updating parameters.

### 📌 Example: YouTube Ads

YouTube must decide in milliseconds which ad to show to a user. This requires fast, scalable machine learning algorithms.

---

## Why Traditional Gradient Descent Fails on Big Data

Gradient Descent (GD) requires computing gradients over all data points before making an update.

### Problem: If the dataset has billions of examples, this is too slow.

#### 📌 Computation Cost Example

Suppose each gradient computation takes 1ms:

- **1,000** data points → **1 second** (manageable)
- **10 million** data points → **2.8 hours** (too slow)
- **10 billion** data points → **115.7 days** (impossible!)

**Solution?** We need an algorithm that updates faster and doesn’t require scanning the full dataset every time.

---

## Stochastic Gradient Descent (SGD) – A Game Changer

SGD updates parameters more frequently by using only one data point at a time instead of the entire dataset.

Instead of computing exact gradients, SGD approximates them using small, random samples.

### How SGD Works

1. Pick a random data point.
2. Compute its gradient.
3. Update the model’s parameters.
4. Repeat for the next random data point.
5. Continue until convergence.

### Why is this better?

✔ Much faster updates (real-time processing possible).
✔ Scales well to massive datasets.
✔ Works even if data is streaming.
✔ Allows training complex models efficiently (e.g., deep learning).

### Trade-offs of SGD

❌ Noisy updates (since it's based on a single data point).
❌ Oscillations around the optimal solution.
❌ More sensitive to hyperparameters (step size, learning rate).

---

## Comparing Gradient Descent vs. Stochastic Gradient Descent

| Method | Pros | Cons |
|------------|----------------------------------|--------------------------------|
| **Batch Gradient Descent** | Stable convergence, exact gradients | Very slow on big data |
| **Stochastic Gradient Descent (SGD)** | Faster updates, handles big data | Noisy updates, oscillates around the solution |

📌 **Key Insight:** SGD is almost always faster for large datasets.

---

## The Role of Online Learning

- **Traditional ML (Batch Learning):** Train on a fixed dataset.
- **Online Learning:** Continuously updates the model as new data arrives.

### 📌 Example: Online Learning in Ad Targeting

1. A user visits a webpage.
2. The system predicts which ad the user will click.
3. The user clicks (or doesn’t) on an ad.
4. The model immediately updates based on this feedback.

✔ Always up-to-date with the latest trends.
✔ Can handle rapidly changing data streams (e.g., stock prices, social media).
❌ More difficult to implement and tune.

---

## Practical Challenges with SGD & Online Learning

### Shuffling Data is Crucial

- If training data is sorted (e.g., all negative examples first), SGD may learn bad patterns.
- **Solution:** Shuffle data before training.

### Choosing the Right Learning Rate (Step Size)

- **Too small** → Slow learning.
- **Too large** → Model oscillates and never converges.
- **Solution:** Use a **decaying learning rate**:

\[
\eta_t = \frac{\eta_0}{1+t}
\]

✔ Starts with large updates.
✔ Gradually reduces update size over time.

### SGD Doesn't Fully Converge (Oscillations)

- Unlike gradient descent, SGD never truly stops.
- Instead, it bounces around the optimal solution.
- **Solution:** Averaging the last few iterations for stability.

### Mini-Batch SGD: A Compromise

- Instead of **1 data point per update**, use **small batches** (e.g., 32 or 128).
- Reduces noise while keeping updates fast.
- **Used heavily in Deep Learning (Neural Networks).**

---

## Distributed & Parallel Machine Learning

Big datasets require big compute power.

### Techniques to scale ML to massive datasets:

- **GPUs & TPUs** (used for Deep Learning).
- **Parallel Processing** (splitting data across multiple machines).
- **Distributed ML frameworks** (e.g., TensorFlow, PyTorch, Apache Spark MLlib).

### 📌 Example: Google Search

- Processes **trillions** of queries.
- Uses **distributed machine learning** to rank results in milliseconds.

