# Table of Contents

1. [Course Overview](#course-overview)
2. [Nearest Neighbor Search: The Basics](#1-nearest-neighbor-search-the-basics)
    - [(A) Data Representation: How Do We Represent a Document?](#a-data-representation-how-do-we-represent-a-document)
    - [(B) Distance Metrics: How Do We Compare Documents?](#b-distance-metrics-how-do-we-compare-documents)
3. [Critical Components of Nearest Neighbor Search](#2-critical-components-of-nearest-neighbor-search)
4. [The Complexity of Nearest Neighbor Search](#3-the-complexity-of-nearest-neighbor-search)
5. [Speeding Up Nearest Neighbor Search](#4-speeding-up-nearest-neighbor-search)
    - [(A) KD-Trees (Efficient Search for Low/Medium Dimensions)](#a-kd-trees-efficient-search-for-lowmedium-dimensions)
    - [(B) Approximate Nearest Neighbor (ANN) with Locality-Sensitive Hashing (LSH)](#b-approximate-nearest-neighbor-ann-with-locality-sensitive-hashing-lsh)
6. [Introduction to Clustering](#5-introduction-to-clustering)
7. [Clustering as an Unsupervised Learning Task](#6-clustering-as-an-unsupervised-learning-task)
8. [K-Means Clustering](#7-k-means-clustering)
    - [Basic Algorithm](#basic-algorithm)
    - [Key Properties](#key-properties)
    - [K-Means++ (Smart Initialization)](#k-means-smart-initialization)
9. [Evaluating Clustering Quality & Choosing k](#8-evaluating-clustering-quality--choosing-k)
    - [Cluster Heterogeneity](#cluster-heterogeneity)
    - [Choosing k](#choosing-k)
10. [Parallelizing K-Means Using MapReduce](#9-parallelizing-k-means-using-mapreduce)
    - [MapReduce Framework](#mapreduce-framework)
    - [Optimizations](#optimizations)
11. [Applications of Clustering](#10-applications-of-clustering)
12. [Overview](#overview)
13. [Why Probabilistic Clustering?](#why-probabilistic-clustering)
14. [Mixture Models & Soft Assignments](#mixture-models--soft-assignments)
    - [Limitations of K-Means](#limitations-of-k-means)
    - [Mixture Models Approach](#mixture-models-approach)
15. [Gaussian Mixture Model (GMM)](#gaussian-mixture-model-gmm)
    - [Key Equations in GMM](#key-equations-in-gmm)
16. [Expectation-Maximization (EM) Algorithm](#expectation-maximization-em-algorithm)
    - [Steps of the EM Algorithm](#steps-of-the-em-algorithm)
    - [Mathematical Form of the EM Steps](#mathematical-form-of-the-em-steps)
    - [Intuition Behind EM](#intuition-behind-em)
17. [Comparison: GMM vs. K-Means](#comparison-gmm-vs-k-means)
    - [K-Means as a Special Case of GMM](#k-means-as-a-special-case-of-gmm)
18. [Challenges & Practical Considerations](#challenges--practical-considerations)
    - [Convergence & Initialization](#convergence--initialization)
    - [Degeneracy & Overfitting](#degeneracy--overfitting)
    - [Computational Complexity](#computational-complexity)
19. [Motivation for Mixed Membership Models](#motivation-for-mixed-membership-models)
20. [Clustering vs. Mixed Membership Models](#clustering-vs-mixed-membership-models)
    - [Clustering Models (e.g., K-means, Gaussian Mixture Models)](#clustering-models-eg-k-means-gaussian-mixture-models)
    - [Mixed Membership Models (e.g., LDA)](#mixed-membership-models-eg-lda)
21. [Bag-of-Words Representation & Mixture of Multinomials](#bag-of-words-representation--mixture-of-multinomials)
    - [Alternative to Mixture of Gaussians](#alternative-to-mixture-of-gaussians)
22. [Latent Dirichlet Allocation (LDA)](#latent-dirichlet-allocation-lda)
    - [Key Components of LDA](#key-components-of-lda)
    - [Comparison to Clustering](#comparison-to-clustering)
23. [Inference in LDA: Learning Topics from Data](#inference-in-lda-learning-topics-from-data)
24. [Expectation-Maximization (EM) vs. Gibbs Sampling](#expectation-maximization-em-vs-gibbs-sampling)
    - [Bayesian Approach to LDA](#bayesian-approach-to-lda)
25. [Gibbs Sampling for LDA](#gibbs-sampling-for-lda)
    - [What is Gibbs Sampling?](#what-is-gibbs-sampling)
    - [Steps of Gibbs Sampling in LDA](#steps-of-gibbs-sampling-in-lda)
26. [Collapsed Gibbs Sampling](#collapsed-gibbs-sampling)
    - [Optimization Trick: Marginalizing Out Model Parameters](#optimization-trick-marginalizing-out-model-parameters)
27. [Applications of LDA](#applications-of-lda)
28. [Nearest Neighbor Search & Retrieval](#nearest-neighbor-search--retrieval)
29. [Nearest Neighbor Search (NNS)](#nearest-neighbor-search-nns)
    - [Key Challenges & Solutions](#key-challenges--solutions)
        - [Data Representation](#data-representation)
        - [Distance Metrics](#distance-metrics)
        - [Scalability Issues](#scalability-issues)
    - [Efficient Retrieval Techniques](#efficient-retrieval-techniques)
        - [KD-Trees](#kd-trees)
        - [Locality-Sensitive Hashing (LSH)](#locality-sensitive-hashing-lsh)
30. [Clustering Algorithms](#clustering-algorithms)
    - [1. K-Means Clustering](#1-k-means-clustering)
        - [Iterative Algorithm](#iterative-algorithm)
    - [2. Gaussian Mixture Models (GMMs)](#2-gaussian-mixture-models-gmms)
        - [Key Features](#key-features)
    - [3. Probabilistic Clustering: Mixture Models](#3-probabilistic-clustering-mixture-models)
    - [4. Latent Dirichlet Allocation (LDA) – Topic Modeling](#4-latent-dirichlet-allocation-lda--topic-modeling)
        - [Why Mixed Membership Models?](#why-mixed-membership-models)
        - [LDA Components](#lda-components)
        - [LDA Inference: Gibbs Sampling](#lda-inference-gibbs-sampling)
        - [Applications](#applications)
    - [5. Hierarchical Clustering](#5-hierarchical-clustering)
        - [Why Use Hierarchical Clustering?](#why-use-hierarchical-clustering)
        - [Two Main Types](#two-main-types)
            - [Divisive Clustering (Top-Down)](#divisive-clustering-top-down)
            - [Agglomerative Clustering (Bottom-Up)](#agglomerative-clustering-bottom-up)
        - [Linkage Methods](#linkage-methods)
        - [Cutting the Dendrogram](#cutting-the-dendrogram)
    - [6. Hidden Markov Models (HMMs)](#6-hidden-markov-models-hmms)
        - [Why Use HMMs?](#why-use-hmms)
        - [HMM Components](#hmm-components)
        - [Inference in HMMs](#inference-in-hmms)
        - [Applications](#applications)

# Course Overview

This course is part of a machine learning specialization designed to be taken in sequence. It focuses on two key concepts: clustering and retrieval, both widely used in practical applications.

- **Retrieval:** Finding similar items (e.g., recommending similar products, suggesting related news articles, matching social media users).
- **Clustering:** Grouping data into meaningful categories (e.g., segmenting customers, detecting topic clusters in documents, categorizing images).

Unlike previous courses in the specialization, which covered regression and classification, this course focuses on unsupervised learning techniques.


## 1. Nearest Neighbor Search: The Basics

**Concept:**

Given a query document, find the most similar document in a dataset.
Instead of scanning all documents (brute-force), we structure the search to make it efficient.

**Formulation:**

- **1-Nearest Neighbor (1-NN):** Find the single most similar document.
- **k-Nearest Neighbors (k-NN):** Retrieve the top-k most similar documents.

**Algorithm for 1-NN:**

1. Compute distances from the query document to all other documents.
2. Identify the document with the smallest distance (nearest neighbor).

**Algorithm for k-NN:**

1. Compute distances for all documents.
2. Maintain a sorted list of the k closest neighbors.

## 2. Critical Components of Nearest Neighbor Search

### (A) Data Representation: How Do We Represent a Document?

- **Bag of Words (BoW):** A vector of word counts per document.
- **TF-IDF (Term Frequency - Inverse Document Frequency):**
  - Gives higher importance to rare words in a document.
  - Downweights common words like "the," "and," etc.
  - Helps refine document similarity calculations.

### (B) Distance Metrics: How Do We Compare Documents?

- **Euclidean Distance:** Measures absolute distance in high-dimensional space (not ideal for text).
- **Cosine Similarity:** Measures the angle between two document vectors (better for text comparison).
  - **Key Advantage:** Invariant to document length.
  - **Trade-off:** It may make short and long documents seem equally similar when they aren't.
- **Hybrid Approaches:** Different distance metrics for different features (e.g., cosine similarity for text, Euclidean for numerical features like view counts).

## 3. The Complexity of Nearest Neighbor Search

- **Brute-force search (Linear Scan):**
  - **1-NN:** $ O(N) $ (N = number of documents).
  - **k-NN:** $ O(N \log k) $ (if implemented with an efficient priority queue).
  - **Problem:** Too slow for large datasets (millions/billions of documents).

## 4. Speeding Up Nearest Neighbor Search

### (A) KD-Trees (Efficient Search for Low/Medium Dimensions)

A binary tree that recursively partitions data along dimensions.

**Steps to build a KD-tree:**

1. Choose a splitting dimension (e.g., word frequency).
2. Split the dataset at the median value along that dimension.
3. Recursively partition data into subspaces.

**KD-tree Search Process:**

1. Start at the root and traverse to the leaf containing the query.
2. Compare distances within the nearest partition.
3. Backtrack and prune partitions that can't contain a closer neighbor.

**Complexity:**

- **Best case:** $ O(\log N) $ (highly structured data).
- **Worst case:** $ O(N) $ (bad partitioning).

**Issues with KD-Trees:**

- **High-dimensional failure:**
  - In high dimensions, most points are far apart, leading to exponential complexity in $ d $ (curse of dimensionality).
  - **Rule of thumb:** Only useful if $ N >> 2^d $.

**Solution:** Approximate Nearest Neighbor search.

### (B) Approximate Nearest Neighbor (ANN) with Locality-Sensitive Hashing (LSH)

**Key Idea:** Instead of exact nearest neighbors, find a “good enough” neighbor quickly.

**LSH Process:**

1. Randomly project data onto multiple random hyperplanes.
2. Assign each point a binary bit-vector based on which side of the hyperplane it falls.
3. Store these bit-vectors in a hash table (faster lookup).
4. For a query, search its hashed bucket and nearby bins.

**Trade-off:**

- Much faster than brute force or KD-trees in high dimensions.
- Less accuracy, but we control the speed vs. accuracy trade-off.

**Advantages of LSH:**

- Scales well to large datasets.
- Works better than KD-trees in high dimensions.
- Has probabilistic guarantees on search quality.

# Course Overview

This course is part of a machine learning specialization designed to be taken in sequence. It focuses on two key concepts: clustering and retrieval, both widely used in practical applications.

- **Retrieval:** Finding similar items (e.g., recommending similar products, suggesting related news articles, matching social media users).
- **Clustering:** Grouping data into meaningful categories (e.g., segmenting customers, detecting topic clusters in documents, categorizing images).

Unlike previous courses in the specialization, which covered regression and classification, this course focuses on unsupervised learning techniques.

## 1. Nearest Neighbor Search: The Basics

**Concept:**

Given a query document, find the most similar document in a dataset.
Instead of scanning all documents (brute-force), we structure the search to make it efficient.

**Formulation:**

- **1-Nearest Neighbor (1-NN):** Find the single most similar document.
- **k-Nearest Neighbors (k-NN):** Retrieve the top-k most similar documents.

**Algorithm for 1-NN:**

1. Compute distances from the query document to all other documents.
2. Identify the document with the smallest distance (nearest neighbor).

**Algorithm for k-NN:**

1. Compute distances for all documents.
2. Maintain a sorted list of the k closest neighbors.

## 2. Critical Components of Nearest Neighbor Search

### (A) Data Representation: How Do We Represent a Document?

- **Bag of Words (BoW):** A vector of word counts per document.
- **TF-IDF (Term Frequency - Inverse Document Frequency):**
  - Gives higher importance to rare words in a document.
  - Downweights common words like "the," "and," etc.
  - Helps refine document similarity calculations.

### (B) Distance Metrics: How Do We Compare Documents?

- **Euclidean Distance:** Measures absolute distance in high-dimensional space (not ideal for text).
- **Cosine Similarity:** Measures the angle between two document vectors (better for text comparison).
  - **Key Advantage:** Invariant to document length.
  - **Trade-off:** It may make short and long documents seem equally similar when they aren't.
- **Hybrid Approaches:** Different distance metrics for different features (e.g., cosine similarity for text, Euclidean for numerical features like view counts).

## 3. The Complexity of Nearest Neighbor Search

- **Brute-force search (Linear Scan):**
  - **1-NN:** $ O(N) $ (N = number of documents).
  - **k-NN:** $ O(N \log k) $ (if implemented with an efficient priority queue).
  - **Problem:** Too slow for large datasets (millions/billions of documents).

## 4. Speeding Up Nearest Neighbor Search

### (A) KD-Trees (Efficient Search for Low/Medium Dimensions)

A binary tree that recursively partitions data along dimensions.

**Steps to build a KD-tree:**

1. Choose a splitting dimension (e.g., word frequency).
2. Split the dataset at the median value along that dimension.
3. Recursively partition data into subspaces.

**KD-tree Search Process:**

1. Start at the root and traverse to the leaf containing the query.
2. Compare distances within the nearest partition.
3. Backtrack and prune partitions that can't contain a closer neighbor.

**Complexity:**

- **Best case:** $ O(\log N) $ (highly structured data).
- **Worst case:** $ O(N) $ (bad partitioning).

**Issues with KD-Trees:**

- **High-dimensional failure:**
  - In high dimensions, most points are far apart, leading to exponential complexity in $ d $ (curse of dimensionality).
  - **Rule of thumb:** Only useful if $ N >> 2^d $.

**Solution:** Approximate Nearest Neighbor search.

### (B) Approximate Nearest Neighbor (ANN) with Locality-Sensitive Hashing (LSH)

**Key Idea:** Instead of exact nearest neighbors, find a “good enough” neighbor quickly.

**LSH Process:**

1. Randomly project data onto multiple random hyperplanes.
2. Assign each point a binary bit-vector based on which side of the hyperplane it falls.
3. Store these bit-vectors in a hash table (faster lookup).
4. For a query, search its hashed bucket and nearby bins.

**Trade-off:**

- Much faster than brute force or KD-trees in high dimensions.
- Less accuracy, but we control the speed vs. accuracy trade-off.

**Advantages of LSH:**

- Scales well to large datasets.
- Works better than KD-trees in high dimensions.
- Has probabilistic guarantees on search quality.

## 5. Introduction to Clustering

Unlike traditional document retrieval, which finds similar documents based on input queries, clustering is about discovering structure in data.
Goal: Group similar documents (or data points) without predefined labels.
Example: Articles about sports, world news, or entertainment are grouped automatically without explicit category labels.

## 6. Clustering as an Unsupervised Learning Task

- **Supervised Learning:** Has labeled training data (e.g., house price prediction, sentiment analysis).
- **Unsupervised Learning (Clustering):** No labels; the algorithm must discover structure.

**Example:** Given word count features of documents, the model groups them based on similarity.

**Cluster Representation:**

- Each cluster is defined by its centroid (mean position) and its shape (e.g., ellipses).
- Assignment is based on distance to cluster centroids.
- The process is iterative—assign points → update centroids → repeat.

## 7. K-Means Clustering

### Basic Algorithm

1. Initialize k cluster centers randomly.
2. Assign each data point to the closest cluster (using Euclidean distance).
3. Update cluster centroids to be the mean of points in the cluster.
4. Repeat steps 2 & 3 until convergence.

### Key Properties

- **Centroid-Based:** Uses mean to determine cluster centers.
- **Partitioning Method:** Each data point is assigned to exactly one cluster.
- **Sensitive to Initialization:** Poor starting points lead to suboptimal solutions.

### K-Means++ (Smart Initialization)

Instead of random initialization, K-Means++:
- Picks the first centroid randomly.
- Selects subsequent centroids proportional to squared distance from existing centroids.
- Ensures better separation of clusters.
- Leads to faster convergence and better clustering.

## 8. Evaluating Clustering Quality & Choosing k

### Cluster Heterogeneity

Sum of squared distances between points and their assigned centroid.
Lower heterogeneity = better clustering.

### Choosing k:

- **Elbow Method:** Plot heterogeneity vs. k. Choose k at the point where adding more clusters results in diminishing returns.
  - Too small k: Overly broad clusters.
  - Too large k: Overfitting (clusters too small to be meaningful).

## 9. Parallelizing K-Means Using MapReduce

Since K-Means needs to process large-scale data, we can distribute it using MapReduce.

### MapReduce Framework

- **Map Phase (Classification Step):**
  - Assigns each data point to the closest cluster center.
  - Emits (cluster label, data point) pairs.
- **Reduce Phase (Recenter Step):**
  - Aggregates data points for each cluster.
  - Computes new centroids (mean of points in each cluster).

**Iterative Process:** Since K-Means is an iterative algorithm, MapReduce needs to be run multiple times until convergence.

### Optimizations:

- **Combiner Step:** Local aggregation before reducing, reducing network communication.
- **Efficient Data Partitioning:** Ensure balanced workload across machines.

## 10. Applications of Clustering

Clustering is widely used across industries:

1. **Information Retrieval**
   - Google News: Groups similar articles together.
   - Search Engines: Clusters documents to improve recommendations.
2. **Image Clustering**
   - Google Images Search: Clusters similar images.
   - Ambiguous Queries (e.g., "Cardinal"): Groups images into categories (bird, baseball team, religious figure).
3. **Healthcare & Medicine**
   - Patient Segmentation: Identifies subgroups with similar medical conditions.
   - Seizure Classification: Groups seizure types for better treatment.
4. **E-commerce & Recommendations**
   - Amazon Product Clustering: Groups products based on purchase history.
   - User Clustering: Groups users with similar buying behavior for personalized recommendations.
5. **Crime Forecasting & Housing Prices**
   - Crime Prediction: Clusters geographic regions with similar crime patterns to improve forecasting.
   - Housing Market Analysis: Groups similar neighborhoods for better price predictions.

# Overview

The lecture extends K-means clustering by introducing probabilistic model-based clustering, specifically Mixture Models and the Expectation-Maximization (EM) algorithm. The motivation is to address K-means' limitations, such as:

- Hard assignments of data points to clusters
- Assumption of equal-sized, spherical clusters
- Inability to handle overlapping or elongated clusters

## Why Probabilistic Clustering?

- Real-world data is often not clearly separable.
- Some points may belong to multiple clusters with varying probabilities.
- K-means ignores cluster shapes and assumes equal importance for all dimensions.

## Mixture Models & Soft Assignments

### Limitations of K-Means

- **Hard assignments:** A point must belong to one cluster, ignoring uncertainty.
- **Fixed cluster shapes:** K-means assumes all clusters are equally spread.
- **Inefficiency in overlapping clusters:** K-means cannot express confidence levels in assignments.

### Mixture Models Approach

A Mixture Model allows for soft assignments, meaning each data point is assigned a probability of belonging to each cluster. This is particularly useful in:

- Document clustering, where articles may belong to multiple topics.
- Image clustering, where images might share characteristics with multiple categories.

**Example:**
An article could have:
- 54% probability of belonging to the "World News" cluster
- 45% probability of belonging to the "Science" cluster
- 1% probability of belonging to "Sports"

## Gaussian Mixture Model (GMM)

A Mixture of Gaussians assumes that each cluster follows a Gaussian (Normal) distribution. Instead of just defining cluster centers (like in K-means), we define:

- **Mean (μ):** The center of the Gaussian cluster.
- **Covariance matrix (Σ):** Defines the shape, spread, and orientation of the cluster.
- **Cluster weight (π):** Probability that a randomly selected data point belongs to a given cluster.

This allows GMM to model elliptical and overlapping clusters rather than just spherical ones.

### Key Equations in GMM

- **Cluster probability (prior probability):**
  $
  P(Z_i = k) = \pi_k
  $
  where $\pi_k$ represents the weight of cluster $k$.

- **Likelihood of data given a cluster:**
  $
  P(X_i \mid Z_i = k) = N(X_i \mid \mu_k, \Sigma_k)
  $
  where $\mu_k$ and $\Sigma_k$ define the cluster distribution.

- **Bayes Rule for Soft Assignments (Responsibilities):**
  $
  P(Z_i = k \mid X_i) = \frac{\pi_k N(X_i \mid \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j N(X_i \mid \mu_j, \Sigma_j)}
  $
  This equation determines how much responsibility each cluster takes for a data point.

## Expectation-Maximization (EM) Algorithm

Since both cluster assignments and parameters (means, covariances, weights) are unknown, we use the Expectation-Maximization (EM) algorithm to estimate them iteratively.

### Steps of the EM Algorithm

1. Initialize cluster parameters (randomly or via K-means).
2. **E-Step (Expectation Step):** Compute the responsibilities (soft assignments) using the current parameters.
3. **M-Step (Maximization Step):** Recompute cluster parameters ($\pi$, $\mu$, $\Sigma$) using the soft assignments.
4. Repeat until convergence (i.e., parameters stabilize).

### Mathematical Form of the EM Steps

- **E-Step:** Compute responsibilities using Bayes' rule:
  $
  r_{ik} = \frac{\pi_k N(X_i \mid \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j N(X_i \mid \mu_j, \Sigma_j)}
  $

- **M-Step:** Update cluster parameters:
  - **New cluster weights ($\pi$):**
    $
    \pi_k = \frac{1}{N} \sum_{i=1}^{N} r_{ik}
    $
  - **New cluster means ($\mu$):**
    $
    \mu_k = \frac{\sum_{i=1}^{N} r_{ik} X_i}{\sum_{i=1}^{N} r_{ik}}
    $
  - **New covariance matrices ($\Sigma$):**
    $
    \Sigma_k = \frac{\sum_{i=1}^{N} r_{ik} (X_i - \mu_k)(X_i - \mu_k)^T}{\sum_{i=1}^{N} r_{ik}}
    $

### Intuition Behind EM

- The **E-Step** estimates how likely each point belongs to each cluster.
- The **M-Step** updates cluster parameters based on these probabilities.
- Iteratively refines clusters until convergence.

## Comparison: GMM vs. K-Means

| Feature                  | K-Means                          | Gaussian Mixture Model (GMM)       |
|--------------------------|----------------------------------|------------------------------------|
| Cluster Assignments      | Hard (one cluster per point)     | Soft (probabilistic)               |
| Cluster Shape            | Spherical (equal variance)       | Elliptical (covariance accounts for shape) |
| Handles Overlapping Clusters? | No                           | Yes                                |
| Probability Estimates?   | No                               | Yes                                |
| Optimization Method      | Lloyd’s Algorithm (iterative)    | Expectation-Maximization (EM)      |

### K-Means as a Special Case of GMM

If we fix the covariance matrices $\Sigma_k$ to be equal and drive variances to zero, GMM reduces to K-means. This explains why K-means is a special case of GMM with simplified assumptions.

## Challenges & Practical Considerations

### Convergence & Initialization

- EM only guarantees local convergence (not a global optimum).
- Good initialization (e.g., K-means++ initialization) improves results.
- Log-likelihood can be monitored to track convergence.

### Degeneracy & Overfitting

- **Collapsing clusters:** A single point may form a cluster with zero variance, causing infinite likelihood.
  - **Solution:** Regularize by adding a small value to covariance matrices (Laplace smoothing).

### Computational Complexity

- GMM is slower than K-means due to matrix inversions in covariance estimation.
- **Trade-off:** Higher computational cost for better clustering flexibility.

# Motivation for Mixed Membership Models

## Clustering vs. Mixed Membership Models

### Clustering Models (e.g., K-means, Gaussian Mixture Models)

- Group similar articles into disjoint clusters.
- Assign each document to a single topic (hard or soft assignment).

**Example:** A document about epileptic events might be classified into science (Cluster 4) or technology (Cluster 2) but not both.

### Mixed Membership Models (e.g., LDA)

- Allow documents to belong to multiple topics with different proportions.
- **Example:** A document could be 40% science, 60% technology.
- Each word in the document can be assigned to a different topic.

This approach is more realistic for tasks like:
- News categorization (one article can belong to multiple sections).
- User preference modeling (users read content from multiple domains).
- Document retrieval (retrieving articles based on topic mixture similarity).

## Bag-of-Words Representation & Mixture of Multinomials

### Alternative to Mixture of Gaussians

Instead of representing documents using tf-idf vectors and modeling them with Gaussian Mixtures, LDA uses:

- **Bag-of-Words Representation:** Treats a document as a multiset (unordered list) of words.
- **Multinomial Distribution:** Each topic is modeled as a probability distribution over words.

**Example:**

| Topic      | Top Words                          |
|------------|------------------------------------|
| Science    | brain, neuron, experiment, physics |
| Technology | model, algorithm, system, network  |
| Sports     | game, player, score, football      |

This approach allows documents to be generated from multiple topics rather than just one.

## Latent Dirichlet Allocation (LDA)

### Key Components of LDA

- **Topic-Specific Vocabulary Distributions (Word Probabilities):** Each topic is represented by a probability distribution over words.
  - **Example:** The word "neuron" is highly probable in the "Science" topic but unlikely in "Sports."
- **Document-Specific Topic Distributions (Topic Proportions):** Each document has a probability distribution over topics.
  - **Example:** A science-technology article might be 70% Science, 30% Technology.
- **Word Assignments (Hidden Variables):** Each word in a document is assigned to a specific topic.
  - **Example:** In a sentence, "The neural network model improved", the word "neural" might be assigned to "Science" and "model" to "Technology."

### Comparison to Clustering

| Feature                        | Clustering (e.g., K-Means) | LDA (Mixed Membership)          |
|--------------------------------|----------------------------|---------------------------------|
| Assignment                     | One topic per document     | Multiple topics per document    |
| Uncertainty                    | No uncertainty captured    | Soft topic proportions per document |
| Word-Level Topic Assignment    | No                         | Yes (each word is assigned a topic) |

## Inference in LDA: Learning Topics from Data

Since we only observe words, LDA needs to infer:
- Topic-word distributions (which words belong to which topics).
- Document-topic distributions (which topics are present in each document).
- Word-topic assignments (which topic each word belongs to).

The challenge is that we don’t know these parameters beforehand. We need inference algorithms to estimate them.

## Expectation-Maximization (EM) vs. Gibbs Sampling

LDA could be solved with EM (Expectation-Maximization), but:
- MLE-based estimation overfits in high-dimensional spaces.
- The E-step becomes intractable due to complex probability distributions.

### Bayesian Approach to LDA

LDA is typically solved using a Bayesian approach, which:
- Accounts for uncertainty in model parameters.
- Regularizes estimates to avoid overfitting.
- Uses Gibbs Sampling instead of EM.

## Gibbs Sampling for LDA

### What is Gibbs Sampling?

A Markov Chain Monte Carlo (MCMC) method that iteratively refines topic assignments. Instead of computing the exact posterior, it samples values from conditional distributions.

- Randomly reassigns each word to a topic, considering:
  - How prevalent the topic is in the document.
  - How likely the topic is to generate the word.

### Steps of Gibbs Sampling in LDA

1. Initialize topic assignments randomly for each word in the corpus.
2. For each word in each document:
   - Remove its current assignment.
   - Compute probability of assigning it to each topic using:
     $
     P(Z_{iw} = k \mid \text{other assignments}) \propto P(\text{topic } k \mid \text{document } i) \times P(\text{word } w \mid \text{topic } k)
     $
   - Sample a new topic assignment from this distribution.
   - Update topic counts.
3. Repeat until convergence or computational budget is exhausted.

**Example: Resampling a Word**

Suppose we are reassigning the word "neuron":
- If the Science topic has many words in this document, $ P(\text{Science} \mid \text{Document}) $ is high.
- If "neuron" appears frequently in Science articles, $ P(\text{"neuron"} \mid \text{Science}) $ is high.
- The new topic is sampled based on the product of these probabilities.

## Collapsed Gibbs Sampling

### Optimization Trick: Marginalizing Out Model Parameters

Instead of sampling the topic-word and document-topic distributions separately, we integrate them out. This simplifies the problem to sampling only the word-topic assignments.

**Benefits:**
- Faster convergence.
- Reduces parameter space.
- More efficient for large-scale data.

**Trade-offs:**
- Collapsed Gibbs Sampling eliminates the need to store large probability distributions.
- But it requires sequential processing, making it harder to parallelize.

## Applications of LDA

1. **Topic Discovery in Large Text Corpora**
   - Uncover hidden themes in news articles, research papers, or social media.
   - **Example:** Analyzing Reddit discussions to identify trending topics.
2. **Document Classification & Tagging**
   - Documents can be classified into multiple categories based on their topic mixture.
   - **Example:** A tech article might be 30% AI, 40% cybersecurity, 30% blockchain.
3. **Information Retrieval & Search**
   - Instead of keyword matching, search engines can retrieve topic-related documents.
   - **Example:** A search for "deep learning" may return AI-related articles.
4. **Personalized Recommendations**
   - User preferences can be modeled as topic distributions.
   - **Example:** Netflix recommending shows based on a user's topic interests.

# Nearest Neighbor Search & Retrieval

## Nearest Neighbor Search (NNS)

- **One-nearest neighbor (1-NN):** Finds the most similar data point.
- **K-nearest neighbors (K-NN):** Returns the K most similar points.

### Key Challenges & Solutions

#### Data Representation

- **TF-IDF for text retrieval:** Balances frequency and rarity.
- **Feature weighting:** Title vs. abstract importance in documents.

#### Distance Metrics

- **Cosine Similarity:** Good for text, ignores magnitude.
- **Euclidean Distance:** Used when raw magnitude matters.
- **Scaled Euclidean Distance:** Adjusts importance of features.

#### Scalability Issues

- **Brute-force search:** $ O(N) $ per query, too slow for large datasets.
- **KD-Trees:** Good for low-dimensional data.
- **Locality-Sensitive Hashing (LSH):** For high-dimensional spaces.

### Efficient Retrieval Techniques

#### KD-Trees

- Partition space hierarchically.
- Prune large sections of space to speed up search.
- Struggles with high-dimensional data.

#### Locality-Sensitive Hashing (LSH)

- Randomly project data points into buckets.
- Finds approximate nearest neighbors faster than KD-Trees.
- Useful for high-dimensional data (text, images, embeddings).

# Clustering Algorithms

## 1. K-Means Clustering

Most widely used hard clustering algorithm.

### Iterative Algorithm

- **Assignment Step:** Assign points to the nearest cluster center.
- **Update Step:** Recompute cluster centers.
- Converges to a local minimum, initialization matters.
- **Weakness:** Assumes spherical clusters.

## 2. Gaussian Mixture Models (GMMs)

Soft clustering alternative to K-means.

### Key Features

- Each cluster is a multivariate Gaussian.
- Uses Expectation-Maximization (EM) algorithm:
  - **E-step:** Compute probabilities of cluster membership.
  - **M-step:** Update cluster parameters.
- Handles overlapping clusters better than K-means.

## 3. Probabilistic Clustering: Mixture Models

- Generalization of GMMs.
- Can model complex data distributions.
- Used in document clustering, image segmentation, and anomaly detection.

## 4. Latent Dirichlet Allocation (LDA) – Topic Modeling

### Why Mixed Membership Models?

- Clustering assumes one label per document, but real-world documents belong to multiple topics.
- LDA assigns multiple topics per document with different probabilities.

### LDA Components

- **Topic-word distributions:** Each topic is a probability distribution over words.
- **Document-topic distributions:** Each document has a distribution over topics.
- **Word-topic assignments:** Each word is assigned to a topic.

### LDA Inference: Gibbs Sampling

- Iteratively reassigns words to topics based on:
  - How prevalent the topic is in the document.
  - How likely the word is under the topic.
- Collapsed Gibbs Sampling marginalizes out parameters, reducing complexity.

### Applications

- **News categorization:** Assign articles to multiple topics.
- **Recommendation systems:** Learn user preferences from topic proportions.
- **Search engines:** Retrieve documents based on topic similarity.

## 5. Hierarchical Clustering

### Why Use Hierarchical Clustering?

- No need to specify number of clusters.
- Creates a hierarchy (dendrogram), allowing clusters at different granularities.
- Can capture complex cluster shapes.

### Two Main Types

#### Divisive Clustering (Top-Down)

- Start with one large cluster, recursively split.
- **Example:** Recursive K-Means.

#### Agglomerative Clustering (Bottom-Up)

- Start with each point as its own cluster.
- Merge closest clusters iteratively.

### Linkage Methods

- **Single Linkage:** Merge clusters based on minimum pairwise distance.
- **Complete Linkage:** Merge based on maximum pairwise distance.
- **Ward's Method:** Minimizes variance within clusters (good for balanced clusters).

### Cutting the Dendrogram

- Choose threshold (D) to determine clusters.
- Application-Specific: Small D → more clusters, large D → fewer clusters.

## 6. Hidden Markov Models (HMMs)

### Why Use HMMs?

- Standard clustering ignores sequential dependencies.
- HMMs model dynamic clustering where the current state depends on the previous state.

### HMM Components

- **Hidden States (Clusters):** Underlying structure (e.g., dance moves of bees 🐝).
- **Emission Probabilities:** Probability of observing a given value from a state.
- **Transition Probabilities:** Probability of moving between states.

### Inference in HMMs

- **Baum-Welch Algorithm (EM for HMMs):** Learns model parameters.
- **Viterbi Algorithm:** Finds most probable state sequence.
- **Forward-Backward Algorithm:** Computes soft assignments.

### Applications

- **Speech recognition:** Segmenting audio into phonemes.
- **Stock market analysis:** Identifying market regimes.
- **Biological sequence modeling:** DNA/protein sequence analysis.