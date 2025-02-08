# Introduction to Regression

Regression assumes a relationship between features (input) and response (output). The objective is to learn a function $ f(X) $ that maps input data to continuous outputs.

**Examples:**
- Predicting salary based on education and experience.
- Estimating house price based on square footage and number of rooms.

1. [Simple Linear Regression](#simple-linear-regression)
2. [Multiple Regression Model](#multiple-regression-model)
3. [Polynomial Regression](#polynomial-regression)
4. [Feature Engineering & Extraction](#feature-engineering--extraction)
5. [Matrix Notation for Multiple Regression](#matrix-notation-for-multiple-regression)
6. [Model Fitting](#model-fitting)
7. [Gradient Descent for Multiple Regression](#gradient-descent-for-multiple-regression)
8. [Coefficient Interpretation](#coefficient-interpretation)
9. [Evaluating Regression Models](#evaluating-regression-models)
    - [Loss Functions: Measuring Error](#1-loss-functions-measuring-error)
    - [Training vs. Generalization Error](#2-training-vs-generalization-error)
    - [Overfitting vs. Underfitting](#3-overfitting-vs-underfitting)
    - [Test Error: Estimating Generalization](#4-test-error-estimating-generalization)
    - [Bias-Variance Tradeoff](#5-bias-variance-tradeoff)
    - [Error Decomposition: Three Sources of Error](#6-error-decomposition-three-sources-of-error)
    - [Training, Validation, and Test Sets](#7-training-validation-and-test-sets)
    - [Model Selection Using a Validation Set](#8-model-selection-using-a-validation-set)
    - [Cross-Validation (When Data is Limited)](#9-cross-validation-when-data-is-limited)
10. [Overfitting & Bias-Variance Tradeoff](#overfitting--bias-variance-tradeoff)
11. [Ridge Regression (L2 Regularization)](#ridge-regression-l2-regularization)
12. [Algorithm for Ridge Regression](#algorithm-for-ridge-regression)
13. [Cross-Validation for Choosing $ \lambda $](#cross-validation-for-choosing--lambda-)
14. [Handling the Intercept Term](#handling-the-intercept-term)
15. [Feature Selection in Regression](#feature-selection-in-regression)
    - [Explicit vs. Implicit Feature Selection](#explicit-vs-implicit-feature-selection)
    - [All Subsets Feature Selection](#1-all-subsets-feature-selection)
    - [Greedy Algorithms for Feature Selection](#2-greedy-algorithms-for-feature-selection)
    - [Regularization-Based Feature Selection (Lasso Regression)](#3-regularization-based-feature-selection-lasso-regression)
    - [Optimization of Lasso: Coordinate Descent](#4-optimization-of-lasso-coordinate-descent)
16. [Nonparametric Regression Approaches](#nonparametric-regression-approaches)
    - [Local vs. Global Fits](#local-vs-global-fits)
    - [K-Nearest Neighbors (k-NN) Regression](#k-nearest-neighbors-k-nn-regression)
    - [Weighted k-NN and Kernel Regression](#weighted-k-nn-and-kernel-regression)
    - [Locally Weighted Regression](#locally-weighted-regression)
17. [Theoretical and Practical Aspects](#theoretical-and-practical-aspects)
    - [k-NN for Classification](#k-nn-for-classification)

# Simple Linear Regression

**Definition:** Models a relationship between a single input feature and output using a straight line:
$Y = w_0 + w_1 X + \epsilon$
where:
- $ w_0 $ is the intercept.
- $ w_1 $ is the slope.
- $ \epsilon $ represents error (residuals).

**Goal:** Find $ w_0 $ and $ w_1 $ that best fit the data.

**Optimization:** Minimize Residual Sum of Squares (RSS).

# Multiple Regression Model

**What is Multiple Regression?**
Instead of just one input feature ($ X $), we use multiple features $ X_1, X_2, \ldots, X_d $.

**General form:**
$ Y = w_0 + w_1 X_1 + w_2 X_2 + \ldots + w_d X_d + \epsilon $
where:
- $ w_0 $ is the intercept.
- $ w_1, w_2, \ldots $ are regression coefficients.
- $ \epsilon $ is the error term.

# Polynomial Regression

Extends linear regression by including higher-order terms of a single feature.

**Example:** Instead of modeling $ Y = w_0 + w_1 X $, we add powers of $ X $, like:
$ Y = w_0 + w_1 X + w_2 X^2 + w_3 X^3 + \cdots + w_p X^p + \epsilon $
This allows the model to capture curvature.

# Feature Engineering & Extraction

Transforming Inputs into Features
Features don’t always have to be raw inputs. They can be functions of inputs.

**Example:**
Instead of using just $ X $ (square footage), use:
$ X, X^2 $ (quadratic), $ \sin(X) $, $ \log(X) $, etc.

**Seasonality Example (Time-Series Modeling):**
Housing prices often fluctuate seasonally. We can model this with sinusoidal features:
$ Y = w_0 + w_1 T + w_2 \sin\left(\frac{2\pi T}{12}\right) + w_3 \cos\left(\frac{2\pi T}{12}\right) + \epsilon $
This accounts for cyclical trends, e.g., higher house prices in summer.

**Applications Beyond Housing:**
- Weather Forecasting: Uses multiple seasonal patterns (daily, yearly).
- Flu Monitoring: Flu cases rise and fall seasonally.
- E-Commerce: Seasonal demand forecasting for products (e.g., winter coats).

# Matrix Notation for Multiple Regression

To handle multiple variables efficiently, we rewrite our equations using matrices.

**Input Matrix (H):** Rows = Observations, Columns = Features.
**Weights Vector (W):** Contains regression coefficients.
**Output Vector (Y):** Contains observed values.

The multiple regression model in matrix form:
$ Y = HW + \epsilon $
where:
- $ H $ is the design matrix.
- $ W $ is the vector of regression coefficients.
- $ \epsilon $ is the error vector.

# Model Fitting

**Closed-Form Solution (Normal Equation):**
Deriving the best weights by setting the gradient of Residual Sum of Squares (RSS) to zero. The optimal weights are given by:
$ W = (H^T H)^{-1} H^T Y $

**Pros:**
- Exact solution.
- Works well for small datasets.

**Cons:**
- Computationally expensive ($ O(d^3) $ complexity).
- Requires matrix inversion, which may be unstable.

# Gradient Descent for Multiple Regression

Alternative to matrix inversion for large datasets. Iteratively updates weights using:
$ W(t+1) = W(t) - \eta \nabla RSS $
where:
- $ \eta $ is the learning rate.
- $ \nabla RSS $ is the gradient:
$ \nabla RSS = -2H^T (Y - HW) $

**Intuition:**
- If the model underestimates the effect of a feature, its coefficient increases.
- If it overestimates, the coefficient decreases.

**Pros:**
- Works for large feature spaces.
- Avoids expensive matrix inversion.

**Cons:**
- Requires careful tuning of learning rate.
- Can get stuck in local minima (though less of a problem for regression).

# Coefficient Interpretation

The coefficient $ w_j $ represents the impact of feature $ X_j $ on the output when all other features are held constant.

**Example:**

Housing price model with square footage and number of bathrooms:
$ Y = w_0 + w_1 (\text{square footage}) + w_2 (\text{bathrooms}) + \epsilon $

- $ w_1 $ = How much the price changes per additional square foot, holding bathrooms constant.
- $ w_2 $ = How much price changes when adding a bathroom, assuming square footage remains fixed.

**Caution:**

Coefficients depend on the context of the model. If we omit important variables, coefficients might be misleading.

# Evaluating Regression Models

## 1. Loss Functions: Measuring Error

In machine learning, we define a loss function to measure how bad our predictions are.

### Types of Loss Functions

1. **Absolute Error (L1 Loss)**
$ L(y, \hat{y}) = |y - \hat{y}| $
Measures the absolute difference between actual and predicted values.
Less sensitive to outliers compared to squared error.

2. **Squared Error (L2 Loss)**
$ L(y, \hat{y}) = (y - \hat{y})^2 $
Penalizes large errors more than absolute error.
Leads to models that prioritize minimizing large deviations.

## 2. Training vs. Generalization Error

### Training Error

The error computed on the same data the model was trained on.
Formula:
$ \text{Training Error} = \frac{1}{N} \sum_{i=1}^{N} L(y_i, \hat{y}_i) $
Problem? It’s often too optimistic since the model has already seen this data.

### Generalization Error

The true error on unseen data.
Cannot be computed exactly because we don’t have access to all possible future data.
Goal: Find a good approximation using test error.

## 3. Overfitting vs. Underfitting

### Overfitting

Model fits training data too well, capturing noise instead of real patterns.
Symptoms:
- Very low training error, but high test error.
- Poor performance on new data.
Cause: Model is too complex.

### Underfitting

Model is too simple to capture patterns in data.
Symptoms:
- High training and test error.
- Fails to represent underlying trends.
Cause: Model lacks capacity.

## 4. Test Error: Estimating Generalization

### Why Do We Need Test Error?

Since generalization error is unknown, we approximate it using test error:
$ \text{Test Error} = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} L(y_i, \hat{y}_i) $
Key Property: Test data must be completely unseen by the model.

## 5. Bias-Variance Tradeoff

Machine learning models suffer from two main types of errors:

### Bias (Error due to assumptions)

If a model is too simple, it makes strong assumptions → High Bias.
Example: Linear model trying to fit non-linear data.

### Variance (Error due to sensitivity)

If a model is too complex, it becomes sensitive to training data variations → High Variance.
Example: High-degree polynomial regression fitting noise.

### Tradeoff

Low Bias + Low Variance = Ideal Model (but difficult to achieve).
We aim for a balance between bias and variance.

## 6. Error Decomposition: Three Sources of Error

For any given prediction, total error can be decomposed into:
$ \text{Total Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error} $
- Bias: Error due to model assumptions.
- Variance: Sensitivity to training data.
- Irreducible Error: Noise inherent in data (cannot be eliminated).
Goal: Find the right model complexity that balances bias and variance.

## 7. Training, Validation, and Test Sets

To properly evaluate models, we split data into three sets:

1. **Training Set**
   Used to train the model.

2. **Validation Set**
   Used to tune hyperparameters (e.g., degree of polynomial).
   Helps in model selection.

3. **Test Set**
   Completely untouched data.
   Used for final performance assessment.

## 8. Model Selection Using a Validation Set

### Why Not Use Test Data for Model Selection?

If we optimize a model based on test data, it’s no longer a fair estimate of generalization.
Solution? Introduce a validation set:
- Train models with different complexities.
- Select the best model based on validation error.
- Finally, evaluate on the test set.

## 9. Cross-Validation (When Data is Limited)

When we don’t have enough data to split into training, validation, and test sets, we use cross-validation.

### K-Fold Cross-Validation:

- Split data into K subsets (folds).
- Train on K-1 folds, test on 1 fold.
- Repeat K times and average results.

# Overfitting & Bias-Variance Tradeoff

Complex models (like high-degree polynomials) have low bias but high variance, making them sensitive to training data. Simpler models have high bias but low variance and fail to capture underlying patterns. The goal is to balance bias and variance for optimal predictive performance.

**Example: Polynomial Regression**
- A quadratic fit may capture general trends.
- A 16-degree polynomial will overfit, showing extreme fluctuations.
- When overfitting occurs, the magnitude of model coefficients increases significantly.

# Ridge Regression (L2 Regularization)

Ridge regression penalizes large coefficients by adding an L2 norm term to the cost function.

The cost function is modified to:
$ RSS(w) + \lambda \sum w_j^2 $

Where:
- RSS(w) = Residual Sum of Squares (fit to data)
- $ \lambda $ (lambda) = Regularization parameter controlling penalty strength

**Effect of $ \lambda $:**
- Large $ \lambda $ → Shrinks coefficients (reducing complexity)
- Small $ \lambda $ → Similar to Least Squares Regression (can overfit)
- $ \lambda = 0 $ → Regular least squares regression
- $ \lambda \to \infty $ → All coefficients approach 0 (very high bias, underfitting)
- $ 0 < \lambda < \infty $ → Balanced complexity

# Algorithm for Ridge Regression

**Closed-Form Solution**
The ridge regression parameters can be computed using:
$ w_{ridge} = (H^T H + \lambda I)^{-1} H^T y $

Where:
- $ H $ = Feature matrix
- $ I $ = Identity matrix
- $ \lambda I $ ensures the inverse exists, even when features are highly correlated or there are more features than samples.

**Gradient Descent Formulation**
Ridge regression can also be solved iteratively using gradient descent.

The weight update rule is:
$ w_j(t+1) = (1 - 2 \eta \lambda) w_j(t) - \eta \frac{\partial RSS}{\partial w_j} $

- The first term shrinks the coefficient.
- The second term updates based on the gradient of RSS.

# Cross-Validation for Choosing $ \lambda $

$ \lambda $ must be tuned automatically to avoid manually testing different values.

**K-Fold Cross-Validation:**
- The dataset is split into K equal parts.
- Each subset is used as a validation set while the others are used for training.
- The error across all folds is averaged to find the best $ \lambda $.

**Leave-One-Out Cross-Validation (LOOCV):**
- Uses each data point one at a time as a validation set.
- More accurate, but computationally expensive.

# Handling the Intercept Term

The ridge penalty shrinks all coefficients, including the intercept $ w_0 $.

**Two possible solutions:**
1. Exclude the intercept from regularization by modifying the identity matrix.
2. Center the data around zero before applying ridge regression (common in practice).

# Feature Selection in Regression

Feature selection is crucial for efficiency, interpretability, and improving model performance. It helps in:

- Reducing computation: If we have an extremely large feature set (e.g., 100 billion features), computation becomes infeasible.
- Improving interpretability: Understanding which features matter most (e.g., in house pricing, not every tiny detail, like having a microwave, is important).

## Explicit vs. Implicit Feature Selection

- **Explicit methods:** Search for the best subset of features using algorithms like all subsets selection and greedy methods.
- **Implicit methods:** Use regularization techniques like Lasso Regression to automatically shrink coefficients and perform feature selection.

### 1. All Subsets Feature Selection

Examines every possible feature combination to find the best subset.

**Procedure:**
1. Start with no features (baseline model).
2. Add one feature at a time, computing training error for each.
3. Select the best one-feature model.
4. Move to two-feature models, pick the best.
5. Continue until all features are included.

**Limitation:**
- Computationally infeasible: If there are $ D $ features, the number of models evaluated is $ 2^D $, which becomes impossible for large $ D $.

### 2. Greedy Algorithms for Feature Selection

To avoid computational cost, we use heuristic approaches:

- **Forward Stepwise Selection:**
  - Start with no features and add one at a time, choosing the best improvement at each step.
  - Limitations: Might miss optimal feature sets because once a feature is included, it stays.

- **Backward Stepwise Selection:**
  - Start with all features and remove them one by one.

- **Hybrid Methods:** Add/remove features dynamically (e.g., stepwise selection with elimination).

**Comparison:**

| Method           | Pros                        | Cons                                      |
|------------------|-----------------------------|-------------------------------------------|
| All subsets      | Finds best model            | Computationally infeasible for large $ D $ |
| Forward Selection| Faster than all subsets     | Might miss the best subset                |
| Backward Selection| Can remove redundant features | Requires fitting full model first         |

### 3. Regularization-Based Feature Selection (Lasso Regression)

Instead of manually searching for the best subset, Lasso (L1 Regularization) automatically shrinks some feature coefficients to exactly zero, removing them from the model.

**Lasso vs. Ridge Regression:**

- **Ridge Regression (L2 penalty):**
  - Shrinks coefficients but doesn’t set them exactly to zero.
  - Works well when all features contribute a little.

- **Lasso Regression (L1 penalty):**
  - Shrinks some coefficients to exactly zero, performing feature selection.
  - Ideal when only a subset of features are relevant.

**Why Does Lasso Work?**
- Uses L1 norm ($|w|$ instead of $w^2$ in Ridge).
- Geometrically, L1 forms a diamond shape constraint, which increases the probability of hitting an axis and setting some coefficients to zero (sparseness).
- Unlike thresholding Ridge regression (which fails due to correlated features), Lasso naturally selects features.

### 4. Optimization of Lasso: Coordinate Descent

Since Lasso lacks a closed-form solution, we solve it using Coordinate Descent, which:

- Updates one coefficient at a time while keeping others fixed.
- Uses a technique called soft-thresholding, which shrinks small coefficients to zero.
- Is computationally efficient and works well in high-dimensional settings.

**Algorithm for Lasso (Coordinate Descent):**
1. Initialize all weights to zero.
2. While not converged:
   - Pick a feature $ j $ and compute its effect ($\rho_j$).
   - Update $ w_j $:
     - If $\rho_j$ is small → Set $ w_j = 0 $.
     - If $\rho_j$ is large → Adjust $ w_j $ based on $\lambda$.

**Choosing Lambda ($\lambda$):**
- $\lambda$ controls sparsity:
  - Small $\lambda$ → Less regularization, more features retained.
  - Large $\lambda$ → More regularization, fewer features retained.
- Select optimal $\lambda$ using cross-validation.

# Nonparametric Regression Approaches

Moving beyond fixed-feature models to more flexible approaches like k-Nearest Neighbors (k-NN) and Kernel Regression.
These methods allow model complexity to grow with data.

## Local vs. Global Fits

Traditional regression fits a single function over the entire dataset.
In contrast, nonparametric approaches allow local adjustments, meaning different regions of the input space can have different behaviors.

### K-Nearest Neighbors (k-NN) Regression

**Concept:** Instead of fitting a global function, k-NN finds the k most similar points in the dataset and averages their values.
- 1-Nearest Neighbor (1-NN): The simplest case, where we predict using the closest data point.

**Distance Metric Matters:**
- The choice of distance (Euclidean, Manhattan, weighted distances) impacts which neighbors are selected.
- A Voronoi diagram can be used to visualize regions where each point is the nearest neighbor.

**Higher-Dimensional Spaces:** k-NN generalizes, but high-dimensional data requires careful handling due to sparse observations (Curse of Dimensionality).

**Key Trade-offs:**
- Too small k (e.g., 1-NN) → High variance, overfits to noise.
- Too large k → High bias, oversmooths trends.

### Weighted k-NN and Kernel Regression

- **Weighted k-NN:** Assigns weights to neighbors based on proximity (closer points contribute more).
- **Kernel Regression:** Extends this idea by applying weights to all observations, not just a fixed number of neighbors.

**Common Kernel Functions:**
- Gaussian Kernel (smoothest)
- Epanechnikov Kernel (faster decay)
- Boxcar Kernel (hard cutoffs)

**Kernel Bandwidth (λ):**
- Small λ → High variance (fits details, risk of overfitting).
- Large λ → High bias (overly smooth, underfits data).

**Choosing λ or k?**
- Cross-validation is typically used to tune these hyperparameters.

### Locally Weighted Regression

Instead of fitting a constant locally (as in kernel regression), we can fit local polynomials (e.g., locally weighted linear regression).
- **Key Benefit:** Reduces boundary effects and improves performance in curved regions.

## Theoretical and Practical# Introduction to Regression

Regression assumes a relationship between features (input) and response (output). The objective is to learn a function $ f(X) $ that maps input data to continuous outputs.

**Examples:**
- Predicting salary based on education and experience.
- Estimating house price based on square footage and number of rooms.

# Simple Linear Regression

**Definition:** Models a relationship between a single input feature and output using a straight line:
$ Y = w_0 + w_1 X + \epsilon $
where:
- $ w_0 $ is the intercept.
- $ w_1 $ is the slope.
- $ \epsilon $ represents error (residuals).

**Goal:** Find $ w_0 $ and $ w_1 $ that best fit the data.

**Optimization:** Minimize Residual Sum of Squares (RSS).

# Multiple Regression Model

**What is Multiple Regression?**
Instead of just one input feature ($ X $), we use multiple features $ X_1, X_2, \ldots, X_d $.

**General form:**
$ Y = w_0 + w_1 X_1 + w_2 X_2 + \ldots + w_d X_d + \epsilon $
where:
- $ w_0 $ is the intercept.
- $ w_1, w_2, \ldots $ are regression coefficients.
- $ \epsilon $ is the error term.

# Polynomial Regression

Extends linear regression by including higher-order terms of a single feature.

**Example:** Instead of modeling $ Y = w_0 + w_1 X $, we add powers of $ X $, like:
$ Y = w_0 + w_1 X + w_2 X^2 + w_3 X^3 + \cdots + w_p X^p + \epsilon $
This allows the model to capture curvature.

# Feature Engineering & Extraction

Transforming Inputs into Features
Features don’t always have to be raw inputs. They can be functions of inputs.

**Example:**
Instead of using just $ X $ (square footage), use:
$ X, X^2 $ (quadratic), $ \sin(X) $, $ \log(X) $, etc.

**Seasonality Example (Time-Series Modeling):**
Housing prices often fluctuate seasonally. We can model this with sinusoidal features:
$ Y = w_0 + w_1 T + w_2 \sin\left(\frac{2\pi T}{12}\right) + w_3 \cos\left(\frac{2\pi T}{12}\right) + \epsilon $
This accounts for cyclical trends, e.g., higher house prices in summer.

**Applications Beyond Housing:**
- Weather Forecasting: Uses multiple seasonal patterns (daily, yearly).
- Flu Monitoring: Flu cases rise and fall seasonally.
- E-Commerce: Seasonal demand forecasting for products (e.g., winter coats).

# Matrix Notation for Multiple Regression

To handle multiple variables efficiently, we rewrite our equations using matrices.

**Input Matrix (H):** Rows = Observations, Columns = Features.
**Weights Vector (W):** Contains regression coefficients.
**Output Vector (Y):** Contains observed values.

The multiple regression model in matrix form:
$ Y = HW + \epsilon $
where:
- $ H $ is the design matrix.
- $ W $ is the vector of regression coefficients.
- $ \epsilon $ is the error vector.

# Model Fitting

**Closed-Form Solution (Normal Equation):**
Deriving the best weights by setting the gradient of Residual Sum of Squares (RSS) to zero. The optimal weights are given by:
$ W = (H^T H)^{-1} H^T Y $

**Pros:**
- Exact solution.
- Works well for small datasets.

**Cons:**
- Computationally expensive ($ O(d^3) $ complexity).
- Requires matrix inversion, which may be unstable.

# Gradient Descent for Multiple Regression

Alternative to matrix inversion for large datasets. Iteratively updates weights using:
$ W(t+1) = W(t) - \eta \nabla RSS $
where:
- $ \eta $ is the learning rate.
- $ \nabla RSS $ is the gradient:
$ \nabla RSS = -2H^T (Y - HW) $

**Intuition:**
- If the model underestimates the effect of a feature, its coefficient increases.
- If it overestimates, the coefficient decreases.

**Pros:**
- Works for large feature spaces.
- Avoids expensive matrix inversion.

**Cons:**
- Requires careful tuning of learning rate.
- Can get stuck in local minima (though less of a problem for regression).

# Coefficient Interpretation

The coefficient $ w_j $ represents the impact of feature $ X_j $ on the output when all other features are held constant.

**Example:**

Housing price model with square footage and number of bathrooms:
$ Y = w_0 + w_1 (\text{square footage}) + w_2 (\text{bathrooms}) + \epsilon $

- $ w_1 $ = How much the price changes per additional square foot, holding bathrooms constant.
- $ w_2 $ = How much price changes when adding a bathroom, assuming square footage remains fixed.

**Caution:**

Coefficients depend on the context of the model. If we omit important variables, coefficients might be misleading.

# Evaluating Regression Models

## 1. Loss Functions: Measuring Error

In machine learning, we define a loss function to measure how bad our predictions are.

### Types of Loss Functions

1. **Absolute Error (L1 Loss)**
$ L(y, \hat{y}) = |y - \hat{y}| $
Measures the absolute difference between actual and predicted values.
Less sensitive to outliers compared to squared error.

2. **Squared Error (L2 Loss)**
$ L(y, \hat{y}) = (y - \hat{y})^2 $
Penalizes large errors more than absolute error.
Leads to models that prioritize minimizing large deviations.

## 2. Training vs. Generalization Error

### Training Error

The error computed on the same data the model was trained on.
Formula:
$ \text{Training Error} = \frac{1}{N} \sum_{i=1}^{N} L(y_i, \hat{y}_i) $
Problem? It’s often too optimistic since the model has already seen this data.

### Generalization Error

The true error on unseen data.
Cannot be computed exactly because we don’t have access to all possible future data.
Goal: Find a good approximation using test error.

## 3. Overfitting vs. Underfitting

### Overfitting

Model fits training data too well, capturing noise instead of real patterns.
Symptoms:
- Very low training error, but high test error.
- Poor performance on new data.
Cause: Model is too complex.

### Underfitting

Model is too simple to capture patterns in data.
Symptoms:
- High training and test error.
- Fails to represent underlying trends.
Cause: Model lacks capacity.

## 4. Test Error: Estimating Generalization

### Why Do We Need Test Error?

Since generalization error is unknown, we approximate it using test error:
$ \text{Test Error} = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} L(y_i, \hat{y}_i) $
Key Property: Test data must be completely unseen by the model.

## 5. Bias-Variance Tradeoff

Machine learning models suffer from two main types of errors:

### Bias (Error due to assumptions)

If a model is too simple, it makes strong assumptions → High Bias.
Example: Linear model trying to fit non-linear data.

### Variance (Error due to sensitivity)

If a model is too complex, it becomes sensitive to training data variations → High Variance.
Example: High-degree polynomial regression fitting noise.

### Tradeoff

Low Bias + Low Variance = Ideal Model (but difficult to achieve).
We aim for a balance between bias and variance.

## 6. Error Decomposition: Three Sources of Error

For any given prediction, total error can be decomposed into:
$ \text{Total Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error} $
- Bias: Error due to model assumptions.
- Variance: Sensitivity to training data.
- Irreducible Error: Noise inherent in data (cannot be eliminated).
Goal: Find the right model complexity that balances bias and variance.

## 7. Training, Validation, and Test Sets

To properly evaluate models, we split data into three sets:

1. **Training Set**
   Used to train the model.

2. **Validation Set**
   Used to tune hyperparameters (e.g., degree of polynomial).
   Helps in model selection.

3. **Test Set**
   Completely untouched data.
   Used for final performance assessment.

## 8. Model Selection Using a Validation Set

### Why Not Use Test Data for Model Selection?

If we optimize a model based on test data, it’s no longer a fair estimate of generalization.
Solution? Introduce a validation set:
- Train models with different complexities.
- Select the best model based on validation error.
- Finally, evaluate on the test set.

## 9. Cross-Validation (When Data is Limited)

When we don’t have enough data to split into training, validation, and test sets, we use cross-validation.

### K-Fold Cross-Validation:

- Split data into K subsets (folds).
- Train on K-1 folds, test on 1 fold.
- Repeat K times and average results.

# Overfitting & Bias-Variance Tradeoff

Complex models (like high-degree polynomials) have low bias but high variance, making them sensitive to training data. Simpler models have high bias but low variance and fail to capture underlying patterns. The goal is to balance bias and variance for optimal predictive performance.

**Example: Polynomial Regression**
- A quadratic fit may capture general trends.
- A 16-degree polynomial will overfit, showing extreme fluctuations.
- When overfitting occurs, the magnitude of model coefficients increases significantly.

# Ridge Regression (L2 Regularization)

Ridge regression penalizes large coefficients by adding an L2 norm term to the cost function.

The cost function is modified to:
$ RSS(w) + \lambda \sum w_j^2 $

Where:
- RSS(w) = Residual Sum of Squares (fit to data)
- $ \lambda $ (lambda) = Regularization parameter controlling penalty strength

**Effect of $ \lambda $:**
- Large $ \lambda $ → Shrinks coefficients (reducing complexity)
- Small $ \lambda $ → Similar to Least Squares Regression (can overfit)
- $ \lambda = 0 $ → Regular least squares regression
- $ \lambda \to \infty $ → All coefficients approach 0 (very high bias, underfitting)
- $ 0 < \lambda < \infty $ → Balanced complexity

# Algorithm for Ridge Regression

**Closed-Form Solution**
The ridge regression parameters can be computed using:
$ w_{ridge} = (H^T H + \lambda I)^{-1} H^T y $

Where:
- $ H $ = Feature matrix
- $ I $ = Identity matrix
- $ \lambda I $ ensures the inverse exists, even when features are highly correlated or there are more features than samples.

**Gradient Descent Formulation**
Ridge regression can also be solved iteratively using gradient descent.

The weight update rule is:
$ w_j(t+1) = (1 - 2 \eta \lambda) w_j(t) - \eta \frac{\partial RSS}{\partial w_j} $

- The first term shrinks the coefficient.
- The second term updates based on the gradient of RSS.

# Cross-Validation for Choosing $ \lambda $

$ \lambda $ must be tuned automatically to avoid manually testing different values.

**K-Fold Cross-Validation:**
- The dataset is split into K equal parts.
- Each subset is used as a validation set while the others are used for training.
- The error across all folds is averaged to find the best $ \lambda $.

**Leave-One-Out Cross-Validation (LOOCV):**
- Uses each data point one at a time as a validation set.
- More accurate, but computationally expensive.

# Handling the Intercept Term

The ridge penalty shrinks all coefficients, including the intercept $ w_0 $.

**Two possible solutions:**
1. Exclude the intercept from regularization by modifying the identity matrix.
2. Center the data around zero before applying ridge regression (common in practice).

# Feature Selection in Regression

Feature selection is crucial for efficiency, interpretability, and improving model performance. It helps in:

- Reducing computation: If we have an extremely large feature set (e.g., 100 billion features), computation becomes infeasible.
- Improving interpretability: Understanding which features matter most (e.g., in house pricing, not every tiny detail, like having a microwave, is important).

## Explicit vs. Implicit Feature Selection

- **Explicit methods:** Search for the best subset of features using algorithms like all subsets selection and greedy methods.
- **Implicit methods:** Use regularization techniques like Lasso Regression to automatically shrink coefficients and perform feature selection.

### 1. All Subsets Feature Selection

Examines every possible feature combination to find the best subset.

**Procedure:**
1. Start with no features (baseline model).
2. Add one feature at a time, computing training error for each.
3. Select the best one-feature model.
4. Move to two-feature models, pick the best.
5. Continue until all features are included.

**Limitation:**
- Computationally infeasible: If there are $ D $ features, the number of models evaluated is $ 2^D $, which becomes impossible for large $ D $.

### 2. Greedy Algorithms for Feature Selection

To avoid computational cost, we use heuristic approaches:

- **Forward Stepwise Selection:**
  - Start with no features and add one at a time, choosing the best improvement at each step.
  - Limitations: Might miss optimal feature sets because once a feature is included, it stays.

- **Backward Stepwise Selection:**
  - Start with all features and remove them one by one.

- **Hybrid Methods:** Add/remove features dynamically (e.g., stepwise selection with elimination).

**Comparison:**

| Method           | Pros                        | Cons                                      |
|------------------|-----------------------------|-------------------------------------------|
| All subsets      | Finds best model            | Computationally infeasible for large $ D $ |
| Forward Selection| Faster than all subsets     | Might miss the best subset                |
| Backward Selection| Can remove redundant features | Requires fitting full model first         |

### 3. Regularization-Based Feature Selection (Lasso Regression)

Instead of manually searching for the best subset, Lasso (L1 Regularization) automatically shrinks some feature coefficients to exactly zero, removing them from the model.

**Lasso vs. Ridge Regression:**

- **Ridge Regression (L2 penalty):**
  - Shrinks coefficients but doesn’t set them exactly to zero.
  - Works well when all features contribute a little.

- **Lasso Regression (L1 penalty):**
  - Shrinks some coefficients to exactly zero, performing feature selection.
  - Ideal when only a subset of features are relevant.

**Why Does Lasso Work?**
- Uses L1 norm ($|w|$ instead of $w^2$ in Ridge).
- Geometrically, L1 forms a diamond shape constraint, which increases the probability of hitting an axis and setting some coefficients to zero (sparseness).
- Unlike thresholding Ridge regression (which fails due to correlated features), Lasso naturally selects features.

### 4. Optimization of Lasso: Coordinate Descent

Since Lasso lacks a closed-form solution, we solve it using Coordinate Descent, which:

- Updates one coefficient at a time while keeping others fixed.
- Uses a technique called soft-thresholding, which shrinks small coefficients to zero.
- Is computationally efficient and works well in high-dimensional settings.

**Algorithm for Lasso (Coordinate Descent):**
1. Initialize all weights to zero.
2. While not converged:
   - Pick a feature $ j $ and compute its effect ($\rho_j$).
   - Update $ w_j $:
     - If $\rho_j$ is small → Set $ w_j = 0 $.
     - If $\rho_j$ is large → Adjust $ w_j $ based on $\lambda$.

**Choosing Lambda ($\lambda$):**
- $\lambda$ controls sparsity:
  - Small $\lambda$ → Less regularization, more features retained.
  - Large $\lambda$ → More regularization, fewer features retained.
- Select optimal $\lambda$ using cross-validation.

# Nonparametric Regression Approaches

Moving beyond fixed-feature models to more flexible approaches like k-Nearest Neighbors (k-NN) and Kernel Regression.
These methods allow model complexity to grow with data.

## Local vs. Global Fits

Traditional regression fits a single function over the entire dataset.
In contrast, nonparametric approaches allow local adjustments, meaning different regions of the input space can have different behaviors.

### K-Nearest Neighbors (k-NN) Regression

**Concept:** Instead of fitting a global function, k-NN finds the k most similar points in the dataset and averages their values.
- 1-Nearest Neighbor (1-NN): The simplest case, where we predict using the closest data point.

**Distance Metric Matters:**
- The choice of distance (Euclidean, Manhattan, weighted distances) impacts which neighbors are selected.
- A Voronoi diagram can be used to visualize regions where each point is the nearest neighbor.

**Higher-Dimensional Spaces:** k-NN generalizes, but high-dimensional data requires careful handling due to sparse observations (Curse of Dimensionality).

**Key Trade-offs:**
- Too small k (e.g., 1-NN) → High variance, overfits to noise.
- Too large k → High bias, oversmooths trends.

### Weighted k-NN and Kernel Regression

- **Weighted k-NN:** Assigns weights to neighbors based on proximity (closer points contribute more).
- **Kernel Regression:** Extends this idea by applying weights to all observations, not just a fixed number of neighbors.

**Common Kernel Functions:**
- Gaussian Kernel (smoothest)
- Epanechnikov Kernel (faster decay)
- Boxcar Kernel (hard cutoffs)

**Kernel Bandwidth (λ):**
- Small λ → High variance (fits details, risk of overfitting).
- Large λ → High bias (overly smooth, underfits data).

**Choosing λ or k?**
- Cross-validation is typically used to tune these hyperparameters.

### Locally Weighted Regression

Instead of fitting a constant locally (as in kernel regression), we can fit local polynomials (e.g., locally weighted linear regression).
- **Key Benefit:** Reduces boundary effects and improves performance in curved regions.

## Theoretical and Practical

## Theoretical and Practical Aspects

- **Nonparametric models scale with data:** More data = better fits.
- **Mean Squared Error (MSE) Convergence:**
  - If data is noise-free, 1-NN error approaches zero with enough data.
  - If data is noisy, k-NN must increase k as data grows to reduce variance.
- **Curse of Dimensionality:**
  - As dimensions grow, data becomes sparse, making nearest-neighbor search inefficient.
  - Dimensionality reduction or careful feature engineering is crucial.

### k-NN for Classification

Instead of averaging outputs, we assign labels based on majority voting among k neighbors.
- **Example:** Spam filtering based on similarity to past labeled emails.
- High k smooths decision boundaries, reducing overfitting.