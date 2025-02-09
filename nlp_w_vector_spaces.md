# Vector Spaces in NLP

Understanding vector representations is crucial for many NLP tasks, from word similarity to machine translation. This module introduces vector spaces in NLP, focusing on how words and documents can be represented as numerical vectors.

# Table of Contents

- [Vector Spaces in NLP](#vector-spaces-in-nlp)
  - [Why Do We Need Vector Spaces?](#why-do-we-need-vector-spaces)
  - [How to Construct Word Vectors?](#how-to-construct-word-vectors)
  - [Measuring Word and Document Similarity](#measuring-word-and-document-similarity)
  - [Word Arithmetic: Using Vectors for Meaning](#word-arithmetic-using-vectors-for-meaning)
  - [Dimensionality Reduction: PCA for Visualizing Word Vectors](#dimensionality-reduction-pca-for-visualizing-word-vectors)
- [Word Translation with Word Vectors](#word-translation-with-word-vectors)
  - [How Can a Machine Learn to Translate?](#how-can-a-machine-learn-to-translate)
  - [Finding the Transformation Matrix (R)](#finding-the-transformation-matrix-r)
  - [Finding Similar Word Vectors with Nearest Neighbors](#finding-similar-word-vectors-with-nearest-neighbors)
  - [Locality-Sensitive Hashing (LSH) for Faster Search](#locality-sensitive-hashing-lsh-for-faster-search)
  - [Document Search with LSH](#document-search-with-lsh)

## Why Do We Need Vector Spaces?

Words can have similar meanings even if they don’t share the same characters.

### Example:
- **"Where are you heading?"** vs. **"Where are you from?"** → Different meaning despite similar words.
- **"What’s your location?"** vs. **"Where are you?"** → Similar meaning despite different words.

Vector spaces help quantify these relationships by encoding words as numbers.

### Applications of Vector Spaces in NLP:
- **Text similarity:** Helps in question answering, paraphrasing, summarization.
- **Semantic relationships:** Detects word dependencies (e.g., "cereal" and "bowl" are related).
- **Information retrieval:** Search engines rank documents based on vector representations.

🚀 **Famous NLP Principle:** “You shall know a word by the company it keeps” — John Firth

(i.e., words appearing in similar contexts tend to have similar meanings.)

---

## How to Construct Word Vectors?

There are two common ways to build vector space representations:

### Co-Occurrence Matrices (Word-by-Word Design)
- Measures how often words appear together within a fixed distance.
- **Example:**
  - Corpus: "Data science is simple. Raw data is useful."
  - Word: "data"
  - If "data" appears within 2 words of "simple", the count is updated.
  - Each word is represented as a vector of co-occurrence counts.

### Word-by-Document Matrix
- Instead of words, we track how often words appear in documents (topic-based).
- **Example:**
  - Word: "data"
  - Occurrences: 500 times in Entertainment, 6620 in Economy, 9320 in Machine Learning.
  - Words like "film" may have high frequency in Entertainment but low in Economy.

🚀 **Key Insight:**
- Vector representations allow document clustering → Similar topics appear closer in vector space.

---

## Measuring Word and Document Similarity

Once words are transformed into vectors, we need a way to compare their similarity.

### Euclidean Distance (Straight-Line Distance)
- Measures how far apart two vectors are.
- **Formula:**
  $
  d(A, B) = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}
  $
- **Example:**
  - "Entertainment" and "Machine Learning" documents have a Euclidean distance of 10,667.

**Issue:**
- Does not account for vector magnitude (size differences).
- A long article vs. a short article about the same topic might be far apart.

### Cosine Similarity (Angle-Based Similarity)
- More effective than Euclidean Distance for comparing word/document vectors.
- **Formula:**
  $
  \text{cosine similarity} = \frac{A \cdot B}{||A|| \times ||B||}
  $
- If cosine similarity ≈ 1, vectors are very similar. If ≈ 0, they are unrelated.
- **Example:**
  - "Agriculture" and "History" might be far apart (large Euclidean distance) but similar (small cosine angle).

🚀 **Key Takeaways:**
- **Euclidean Distance** is good for small-scale problems.
- **Cosine Similarity** is better for comparing large documents or word embeddings.

---

## Word Arithmetic: Using Vectors for Meaning

You can manipulate word vectors using basic arithmetic.

### Example:
- **"USA" + "Washington DC"** → Produces a vector difference that represents the relationship between a country and its capital.
- If we apply the same difference to **"Russia"**, we should get **"Moscow"**.

🚀 **Word Analogies Using Vectors**

- **"King" - "Man" + "Woman" ≈ "Queen"**
- **"Paris" - "France" + "Germany" ≈ "Berlin"**

---

## Dimensionality Reduction: PCA for Visualizing Word Vectors

Word vectors are high-dimensional (hundreds of dimensions). Principal Component Analysis (PCA) helps reduce dimensions while preserving important relationships.

### Example:
- Words like "city" and "town" or "oil" and "gas" appear closer together in lower dimensions.

### PCA Uses:
- **Eigenvalues and Eigenvectors** to extract important features.
- Projects word vectors into a **2D or 3D space** for visualization.

🚀 **Why Use PCA?**
- Helps understand relationships between words.
- Reduces computational cost in NLP tasks.

# Word Translation with Word Vectors

One of the fundamental applications of NLP is machine translation. The goal is to translate an English word (e.g., **hello**) to its French equivalent (**bonjour**) using word embeddings.

---

## How Can a Machine Learn to Translate?

1. Create word vectors for both English and French words.
2. Find a transformation (matrix **R**) that maps English vectors to French vectors.
3. Search for the nearest vector in the French space to get the translation.

### The Role of Word Embeddings

- English and French word embeddings exist in different vector spaces.
- The goal is to find a transformation (**R matrix**) to align these spaces.
- Instead of creating a hardcoded dictionary, we can train the model on a small word set and generalize to unseen words.

### Finding the Transformation Matrix (R)

Given:
- **X** = matrix of English word embeddings.
- **Y** = matrix of French word embeddings.

We optimize **R** using Gradient Descent to minimize:

$
||XR - Y||_F^2
$

(Frobenius norm): Measures the difference between transformed English words and actual French words.

- **Updating R:** Uses gradient descent to minimize this difference.

🚀 **Key Insight:** By learning **R**, we can translate words without needing a direct dictionary!

---

## Finding Similar Word Vectors with Nearest Neighbors

Once we have transformed the English vector into the French space, we need to find the closest French word vector.

### Brute Force K-Nearest Neighbors (KNN)

- Given a word vector, compare it with all words in the French space.
- Find the **k** closest words using **cosine similarity** or **Euclidean distance**.

💡 **Problem?**
- If we have millions of words, this brute-force search is slow.

### A Faster Solution: Hash Tables

- Instead of searching through every word vector, we divide vectors into **buckets**.
- Similar vectors get placed into the same bucket.
- Hashing allows us to search efficiently within a smaller set.

---

## Locality-Sensitive Hashing (LSH) for Faster Search

To speed up nearest neighbor search, we use **Locality-Sensitive Hashing (LSH)**.

### What is Hashing?

- Think of a cupboard with drawers.
- You store similar items in the same drawer.
- Instead of searching through everything, you only check one drawer.

### Basic Hashing

- Each word vector is assigned a **hash value** based on a hash function.
- **Example hash function:**

$
\text{Hash Value} = \text{Word Vector} \mod 10
$

💡 **Problem?**
- This simple method doesn’t guarantee that similar words end up in the same bucket.

---

## Locality-Sensitive Hashing (LSH)

LSH fixes the problem of basic hashing by ensuring that similar words get hashed to the same bucket.

### How Does LSH Work?

1. Divide the vector space using **random hyperplanes** (splitting lines).
2. Assign hash values based on which side of the hyperplane the vector falls.
3. Similar words will likely fall into the same bucket.

🚀 **Example: Finding Nearby Friends**

- Imagine you are visiting San Francisco.
- Instead of checking every friend in the world, you only check the ones in the USA.
- **LSH does the same thing for words**—it reduces the search space to relevant groups.

### Mathematical Foundation of LSH

- Each **hyperplane** divides the space into two regions.
- Compute **dot product** between the word vector and the normal vector of the hyperplane.
  - **Positive dot product** → One side of the plane.
  - **Negative dot product** → Other side of the plane.
- Assign **binary hash values** (0 or 1) based on sign.
- Combine multiple hash values to get a **unique region (bucket)**.

### Why Use LSH?

✅ Much faster than brute-force search.
✅ Efficiently finds **approximate nearest neighbors**.
✅ Trade-off between accuracy and speed.

---

## Document Search with LSH

Another application of nearest neighbor search is **document retrieval**.

### How to Represent Documents as Vectors?

- Words have **vector representations**.
- A document can be represented as **the sum of its word vectors**.

### Finding Similar Documents

**Example Query:** "Can I get a refund?"

1. Convert the query into a **document vector**.
2. Find the **nearest documents** using LSH.

**Possible matches:**
- "What’s your return policy?"
- "May I get my money back?"

🚀 **Why is this useful?**

- LSH allows **fast document retrieval** for search engines, chatbots, and Q&A systems.

