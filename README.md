# README #

README: Cross-Domain Sentiment Classification

27/09/2013
Danushka Bollegala


#Directory structure and details about files
## src/ 
source code for generating various co-occurrence matrices.

## reviews/ 
original Amazon reviews. train and test sub-directories contain the
the reviews that are used respectively for training a cross-domain
sentiment classification method and to test it.

## work/ 
This directory contains the various co-occurrence matrices we
compute for the domains in the dataset. First, there are four
directories named books, dvd, electronics, and kitchen. These
are the four Amazon product categories that are traditionally
used for cross-domain sentiment classification. For each domain
we have 1000 positive labeled reviews (4 or 5 star rated reviews),
1000 negative labeled reviews (1 or 2 star rated reviews) and
some large number of (ca. 10000) unlabeled reviews (3 star reviews).
We set aside 800 (positive) + 800 (negative) + unlabaled data for
training purposes, whereas the remainder 200 (positive) + 200 (negative)
reviews are used for testing purposes. This is a standard benchmark
split and must be followed exactly in order to be able to directly
compare any results obtained with prior methods of cross-domain
sentiment classification. In each directory named after a domain you
will find five files.

test.positive = This file contains the feature vectors representing
the positive labeled reviews for the domain that corresponds to the
name of the directory which it is in. Each line of this file represents
a feature vector. We select unigrams and bigrams from a review and
represent the document using a boolean valued feature vector following
the bag-of-words model. Note that we first lemmatize each word in a review
using the RASP parser. When generating unigram features, we do not select
unigrams that appear in a pre-defined stop word list. However, when
generating bigram features we only remove a bigram if both words of
the bigram are stop words. 

Apart from test.positive we have test.negative (negative feature vectors for testing),
train.positive (positive feature vectors for training),
train.negative (negative feature vectors for training), and
train.unlabeld (unlabeled feature vectors for training) in 
each folder.

Next, you will find 12 sub-directories in the directory "work" such as
"kitchen-books". These directories contain co-occurrence matrices for a
given pair of source (kitchen in this example) and target (books in this example)
domains. Because we have 4 domains, we have 12 possible combinations of
source and target domains. In each of those 12 sub-directories you will find
the following files.

### DI_list
This is the list of pivots (features that appear in both source and target
domains). We compute the following score to measure its appropriateness
as a pivot (DI stands for 'Domain Independent' and is another name for 'pivots').
score(w) = \sum_{x \in S} (P(w,x) * PMI(w,x)) + \sum_{y \in T} (P(w,y) * PMI(w,y))
Here, S is the set of features that appear in the source domain,
T is the set of features that appear in the target domain,
P(w,x) is the joint probability of the pivot w and a feature x (in the source or
in the target for y), and PMI is the pointwise mutual information which is
given by,
PMI(w,x) = \log \frac{P(w,x)}{P(x)P(w)}. All these probabilities are estimated
using frequency counts in the entire corpus (both source and target domain train reviews).
We select the top ranked 500 pivots and write to the DI_list file.
Each line of this file corresponds to a pivot and has the format:
ID feature P score
ID is a unique incremental integer ID of the pivot that acts as the index
of the co-occurrence matrices that are computed later. 
feature is the pivot itself.
The flag 'P' indicates that this is a pivot (i.e. occurs in both source and target domains).
score is the 'score(w)' value computed as described above.

### DS_list
This file contains the domain specific features. We select the top 500 features
from each domain (500 from source and 500 from target) separately according to their
total PMI value with the domain. For example, to select source specific features
we first select the set of features that occur in the source domain but not in the
target domain. We then rank those selected features in the descending order of the
following score:
\sum_{x \in S} (P(w,x) * PMI(w,x)) 
Note that this is the first term in the formula for computing score(w) above.
We then select the top ranked 500 source specific features. We do the same and select
500 target specific features and this 1000 features comprises our list of domain specific
features. The format of the DS_list file is exactly the same as that of the DI_list file
except for tha fact that the flag 'P' is replaced by either 'S' (indicating that this is
a source-specific feature) or by 'T' (indicating that this is a target-specific feature).

### DSxDS.mat
This the co-occurrence matrix for domain specific features. It is in MATLAB mat format. (symmetric and square)

### DIxDI.mat
This the co-occurrence matrix for domain independent features. It is in MATLAB mat format. (symmetric and square)

### DSxDI.mat
This is the co-occurrence matrix between domain specific (in rows) features and domain independent features
(in columns). It is in MATLAB mat format. This is a rectangular and (obviously) assymmetric matrix.

