Author: Polochon_street
Date: 2021-09-14
Title: Auditory metric learning using training triplets

# Auditory metric learning using training triplets

## Introduction

While developping [bliss](https://lelele.io/bliss.html), a program written to build "smart" playlists by putting together similar songs, I've come across the need to find a good distance metric to compare two songs, represented by vectors of floats. Each float represents one aspect (or one part of an aspect) of the song: tempo, timbre, chroma...

One of the most basic distance metric one could use to compute the distance between two songs is the [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance). While this method works quite well in practice, the issue is that it assumes each feature has the same weight as the others. Meaning, in our case, that for example the tempo of a song is as important as its tone when it comes to music similarity.

Of course, this is wrong, and while most experts will agree that a *universal* distance metric regarding songs is not a reasonable concept, I still thought that it would be possible to get an approximation of it, meaning, a distance that would make everyone roughly agree that a playlist has only similar songs.

Note that here we make the assumption that a good playlist is a playlist made of similar songs (that might or might not "drift" into different genres). You might not agree, but this was set as a ground truth to simplify the study.

## Choice of a distance metric

In order to keep the learning somehow simple, I've chosen to use the [Mahalanobis distance](https://en.wikipedia.org/wiki/Mahalanobis_distance) (or generalized quadratic distance):
Let %$M$% be a [symmetric positive semi-definite matrix](https://en.wikipedia.org/wiki/Positive-definite_matrix#Positive_semidefinite) of size %$n \times n$%, %$x$% and %$y$% two vectors of %$\mathbb{R}^n$%, the Mahalanobis distance between %$x$% and %$y$% is defined as: %$$d_M(x, y) = \sqrt{(x - y)^{T}M(x - y)}$$%

All that is left is then to find the coefficients of %$M$% that will yield the best playlists (and yes, if M is the identify matrix, then it will simply be an Euclidean distance).

## Metric learning

As said before, the idea is to make playlists that most people will be happy with, so since it is such a subjective problem, we asked feedback from different people, in the form of an online survey.

People were presented with three different, thirty seconds-long excerpts of songs, and had to point the one that they felt was the most dissimilar compared to the others:
<img src="survey.png">

Each answer gives us the following information: given %$x_1$%, %$x_2$%, and %$x_3$% three songs, if the user picked %$x_2$% as the odd one out, then if %$d_M$% is the distance metric, we know that %$d_M(x_1, x_3) < d_M(x_1, x_2)$% and %$d_M(x_1, x_3) < d_M(x_2, x_3)$%. That is, it tells us that the first and the third song are closer together than they both are to the second song.

We then need to find the matrix %$M$% that respects all these constraints, or as much as possible if it can't fit all.
We can prove that solving this can be seen as a simple minimization problem without constraints (see [this, 4.1](https://lelele.io/thesis.pdf) if you want the detailed explanation).

We show that the function to minimize is, with %$(x_i, x_j, x_k)$% being a training triplet where %$x_k$% is chosen to be the odd one out, %$M = LL^T$, $\mathcal{R}$% being the set of all songs' indexes, %$\Phi$% the [cumulative Gaussian distribution](https://en.wikipedia.org/wiki/Cumulative_distribution_function), and %$\sigma$% a fixed parameter:
 %$$ f(L, x_i, x_j, x_k, \sigma) = -\sum_{i,j,k \in \mathcal{R}} \log \left(\Phi\left(\frac{d_L(x_k, x_i) - d_L(x_i, x_j)}{\sigma^2}\right)\right) $$%

A quick check allows us to see that that the reasoning is roughly correct: to minimize this function correctly, the term in %$log$% needs to be close to one, and the cumulative distribution function gets close to one when %$d_L(x_k, x_i) > d_L(x_j, x_k)$%, which is the condition we try to match.

After a trial-and-error process, we noticed that adding a regularization parameter %$\lambda$% was useful to avoid overfitting, which made the final function: %$$f(L, x_i, x_j, x_k, \sigma, \lambda) = -\sum_{i,j,k \in \mathcal{R}} \log \left(\Phi\left(\frac{d_L(x_k, x_i) - d_L(x_i, x_j)}{\sigma^2}\right)\right) + \lambda \left|\left|L\right|\right|$$%

## Metric learning in practice

We first need to write our various distance functions, and the function we want to minimize (see [Appendix C here](https://lelele.io/thesis.pdf)):


```python
"""
Distance metric and functions to minimize, along with
their respective matrix-wise derivative.
"""
import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import norm as L2_norm


def d(L, x1, x2):
    """Our custom Mahalanobis distance metric."""
    L = L.reshape(len(x1),len(x1))
    sqrd = ((x1 - x2).dot(L.dot(np.transpose(L)))).dot(x1 - x2)
    # If somehow the value is extremely small and ends up negative,
    # correct that.
    if (0 > sqrd) and (sqrd > -1e-10):
        sqrd = np.abs(sqrd)
    return np.sqrt(sqrd)


def grad_d(L, x1, x2):
    """Matrix-wise differentiation o"""
    L = L.reshape(len(x1), len(x2))
    return grad_d_squared(L, x1, x2) / (2 * d(L, x1, x2))


def grad_d_squared(L, x1, x2):
    L = L.reshape(len(x1), len(x1))
    grad = 2*np.outer(x1-x2, x1-x2).dot(L)
    return grad.ravel()


# x3 here is the odd thing
def delta(L, x1, x2, x3, sigma, second_batch=False):
    ret = (d(L, x2, x3) - d(L, x1, x2)) / sigma
    if second_batch:
        ret = (d(L, x1, x3) - d(L, x1, x2)) / sigma
    return ret


def grad_delta(L, x1, x2, x3, sigma, second_batch=False):
    ret = (grad_d(L, x2, x3) - grad_d(L, x1, x2)) / sigma
    if second_batch:
        ret = (grad_d(L, x1, x3) - grad_d(L, x1, x2)) / sigma
    return ret


def p(L, x1, x2, x3, sigma, second_batch=False):
    cdf = norm.cdf(delta(L, x1, x2, x3, sigma, second_batch))
    if cdf == 0:
        print(delta(L, x1, x2, x3, sigma, second_batch))
    return norm.cdf(delta(L, x1, x2, x3, sigma, second_batch))


def grad_p(L, x1, x2, x3, sigma, second_batch=False):
    return (
        norm.pdf(delta(L, x1, x2, x3, sigma, second_batch)) *
        grad_delta(L, x1, x2, x3, sigma, second_batch)
    )


def log_p(L, x1, x2, x3, sigma, second_batch=False):
    return np.log(p(L, x1, x2, x3, sigma, second_batch))


def grad_log_p(L, x1, x2, x3, sigma, second_batch=False):
    return (
        grad_p(L, x1, x2, x3, sigma, second_batch) /
        p(L, x1, x2, x3, sigma, second_batch)
    )


def opti_fun(L, X, sigma, l):
    """Final optimization function, 'f'"""
    batch_1 = -sum (
        np.array([
            log_p(L, x1, x2, x3, sigma)
            for x1, x2, x3 in X
        ])
    )
    batch_2 = -sum(
        np.array([
            log_p(L, x1, x2, x3, sigma, True)
            for x1, x2, x3 in X
        ])
    )
    return batch_1 + batch_2 + l * L2_norm(L)


def grad_opti_fun(L, X, sigma, l):
    """Derivative of the optimization function."""
    batch_1 = (
        -np.sum(
            np.array([
                grad_log_p(L, x1, x2, x3, sigma)
                for x1, x2, x3 in X
            ]),
            0,
        )
    )
    batch_2 = (
        -np.sum(
            np.array([
                grad_log_p(L, x1, x2, x3, sigma, True)
                for x1, x2, x3 in X
            ]),
            0,
        )
    )
    return batch_1 + batch_2 + l * L / L2_norm(L)
```

Using scipy minimization routine, finding the optimized $L$ is just a matter of:


```python
from scipy.optimize import minimize

def optimize(L0, X, sigma2, l):
    """
    Minimize the `opti_fun` function.
    L0 being the initial value for L, sigma2 a constant, l the
    regularization parameter, and X a matrix of training triplets, like
    [[x1, x2, x3], [x4, x5, x6], ...], where the last element is always
    the odd one out.
    """
    l_dim = len(X[0][0])
   
    res = minimize(
        opti_fun,
        L0,
        args=(X, sigma2, l),
        jac=grad_opti_fun
    )
    L = np.reshape(res.x, [l_dim, l_dim])
    return (res.success, L)
```

Now's the time to actually show the dataset, and start the learning.

## Dataset from the real world®

The survey's answers are compiled into [a CSV file](./answer.csv).

The fields are as follow:
* id: the row ID
* song1, song2, song3: the three songs presented to the user (training triplet)
* picked_song: the song deemed as the odd one out
* person_id: the unique ID of the person who answered one or more training triplets
  Useful to draw consistency conclusions.
* updated_date: stamp
* time_spent: the time spent answering

The survey has been done using songs from the [Free Music Archive](https://freemusicarchive.org/), so they can all be redistributed freely. If you wish to download them (to run your own custom analysis on it, or to reproduce this post), it's available [here](https://lelele.io/dataset.7z).

Now, we will assume that each song is represented by a numpy array of floats. You can also download [bliss-rs](https://github.com/Polochon-street/bliss-rs/)' CSV dump of the FMA songs [here](./analysis.csv).

The following code sets up the training triplets:


```python
"""
Make an array of training triplets.

That is, triplets = [[x1, x2, x3], [x4, x5, x6]], with
x3 and x6 being the song pointed as the odd one out by the
answerer. x1, x2, etc, are represented by their respective
coordinates.
"""
import csv
from functools import reduce

songs = {}

with open('analysis.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        songs[row[0]] = np.array(row[1:], dtype='double')

with open('answer.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    triplets = []
    named_triplets = []
    for row in reader:
        triplet = row[1:4]
        picked_song = row[4]
        triplet = [s for s in triplet if s != picked_song] + [picked_song]
        triplets.append([songs[s] for s in triplet])
        named_triplets.append(triplet)
```

Once they training triplets are set up, we can then start optimizing, with the whole dataset and no regularization parameter first, to keep it simple:


```python
X = np.array(triplets)
l_dim = X.shape[2]
sigma2 = 2    
L0 = np.identity(l_dim).ravel()

# lambda = 0, no regularization parameter
res, L = optimize(L0, X, sigma2, 0)

if not res:
    print('Optimization could not be completed successfully.')
```

Now, we've computed an L matrix, which provides us with a distance metric that should perform better than the euclidean distance. We can verify that by evaluating how many distances it preserved wrt. the survey:


```python
def percentage_preserved_distances(L, X):    
    count = 0    
    for x1, x2, x3 in X:    
        d1 = d(L.ravel(), x1, x2) # short distance    
        d2 = d(L.ravel(), x2, x3) # long distance    
        d3 = d(L.ravel(), x1, x3) # long distance    
        if (d1 < d2) and (d1 < d3):    
            count = count + 1    
    return count / len(X)

print(
    'Percentage of preserved distances for random values: {}'
    .format(percentage_preserved_distances(L0, np.random.rand(*X.shape))),
)
print(
    'Percentage of preserved distances without optimization: {}'
    .format(percentage_preserved_distances(L0, X)),
)
print(
    'Percentage of preserved distances with optimization and overfitting: {}'
    .format(percentage_preserved_distances(L, X)),
)
```

    Percentage of preserved distances for random values: 0.3610648918469218
    Percentage of preserved distances without optimization: 0.4525790349417637
    Percentage of preserved distances with optimization and overfitting: 0.562396006655574


Now, we just used the entire dataset as both a training and a testing set, which is... Suboptimal, and will most likely cause overfitting.

Instead, we want to split our dataset between a training and a testing set, and we try again how our optimization performs against a simple euclidean distance, 5 times:


```python
from datetime import datetime
from sklearn.model_selection import KFold, train_test_split

accuracies = []
accuracies_euclidean = []

print('Started at {}'.format(datetime.utcnow()))

kf = KFold(n_splits=5)
for n, (train_index, test_index) in enumerate(kf.split(X), 1):
    X_train, X_test = X[train_index], X[test_index]
    res, L = optimize(L0, X_train, sigma2, 0)
    accuracy = percentage_preserved_distances(L, X_test)
    euclidean_accuracy = percentage_preserved_distances(L0, X_test)
    if not res:
        print('Could not optimize for the {}th fold.'.format(n))
        continue
    accuracies.append(accuracy)
    print(
        'Percentage of preserved distances for the {0:d}th-fold is: '
        '{1:.2f} vs {2:.2f} for the Euclidean accuracy.'
        .format(n, accuracy, euclidean_accuracy)
    )
    accuracies_euclidean.append(euclidean_accuracy)
```

    Started at 2021-09-19 17:02:39.057327
    Percentage of preserved distances for the 1th-fold is: 0.42 vs 0.42 for the Euclidean accuracy.
    Percentage of preserved distances for the 2th-fold is: 0.53 vs 0.46 for the Euclidean accuracy.
    Percentage of preserved distances for the 3th-fold is: 0.42 vs 0.38 for the Euclidean accuracy.
    Percentage of preserved distances for the 4th-fold is: 0.48 vs 0.51 for the Euclidean accuracy.
    Percentage of preserved distances for the 5th-fold is: 0.46 vs 0.49 for the Euclidean accuracy.


The learnt accuracy here is better, but only slightly.

Let's see what happens when we add a regularization parameter:


```python
from datetime import datetime
from sklearn.model_selection import KFold, train_test_split


def optimize_lambdas(X):
    lambdas = [0.01, 0.1, 1, 5]

    accuracies = [[] for _ in lambdas]
    accuracies_euclidean = []

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for i, l in enumerate(lambdas):
        for n, (train_index, test_index) in enumerate(kf.split(X), 1):
            X_train, X_test = X[train_index], X[test_index]
            res, L = optimize(L0, X_train, sigma2, l)
            if not res:
                continue
            accuracy = percentage_preserved_distances(L, X_test)
            accuracies[i].append(accuracy)

    for n, (train_idex, test_index) in enumerate(kf.split(X), 1):
        X_train, X_test = X[train_index], X[test_index]
        accuracies_euclidean.append(
            percentage_preserved_distances(L0, X_test)
        )

    mean_accuracies = np.array(
        [
            np.nanmean(local_accuracies)
            for local_accuracies in accuracies
        ]
    )
    mean_accuracies = mean_accuracies[~np.isnan(mean_accuracies)]
    idx = mean_accuracies.argmax()
    max_accuracy = mean_accuracies[idx]
    l = lambdas[idx]
    print(
        'Mean accuracy for euclidean is: {0:.2f}.'
        .format(np.mean(accuracies_euclidean)),
    )
    print(
        'Best mean accuracy is {0:.2f} for lambda = {1:.2f}\n'
        .format(max_accuracy, l),
    )

    design, test = train_test_split(X, test_size=0.2, shuffle=True, random_state=0)
    res, L = optimize(L0, design, sigma2, l)
    return res, L, design, test


res, L, design, test = optimize_lambdas(X)
if not res:
    print('Error while optimizing the last design set.')
else:    
    print('Final optimization successful.')
    print(
        'Percentage of distance preserved by the trained metric on the test set: {0:.3f} vs. '
        '{1:.3f} for the Euclidean distance metric.'
        .format(
            percentage_preserved_distances(L, test),
            percentage_preserved_distances(L0, test),
        ),
    )
```

    Mean accuracy for euclidean is: 0.45.
    Best mean accuracy is 0.44 for lambda = 0.10
    
    Final optimization successful.
    Percentage of distance preserved by the trained metric on the test set: 0.512 vs. 0.545 for the Euclidean distance metric.


As we can see, the trained metric performs doesn't perform better than the non-trained, Euclidean distance metric. Let's see if we can improve that.

## Small improvement on metric learning

Some training triplets are contradictory: someone might say that given x1, x2, x3, x2 is the song that should be the odd-one out, but someone else would say that x3 is. We indeed have those cases among the answers, which indeed leads to a loss of accuracy of the learnt distance metric. Different handling of these cases are possible: choose the person that took the most time answering as the truth value (assuming that more time means a better answer), pick one of the two at random, or discarding the two altogether. Here, we will use the latter.

The following (somewhat ugly) code removes triplets that are contradictory, and can then perform the same analysis as before on a more coherent dataset:

One can then perform the same analysis as before on a more coherent dataset:


```python
with open('answer.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    named_triplets = []
    for row in reader:
        triplet = row[1:4]
        picked_song = row[4]
        triplet = [s for s in triplet if s != picked_song] + [picked_song]
        named_triplets.append(triplet)

triplets_dict = {}
for triplet in named_triplets:
    triplets_dict[frozenset(triplet)] = triplets_dict.get(frozenset(triplet), []) + [triplet]
for lists in triplets_dict.values():
    if any(x != lists[0] for x in lists):
        for x in lists:
            named_triplets.remove(x)
triplets = [(songs[s1], songs[s2], songs[s3]) for s1, s2, s3 in named_triplets]
X = np.array(triplets)

design, test = train_test_split(X, test_size=0.2, shuffle=True, random_state=0)
res, L = optimize(L0, design, sigma2, 1)

if not res:
    print('Error while optimizing the last design set.')

print('Final optimization successful.')
print(
    'Percentage of distance preserved by the trained metric on the more coherent test set: {0:.3f} vs. '
    '{1:.3f} for the Euclidean distance metric.'
    .format(
        percentage_preserved_distances(L, test),
        percentage_preserved_distances(L0, test),
    ),
)
```

    Final optimization successful.
    Percentage of distance preserved by the trained metric on the more coherent test set: 0.482 vs. 0.465 for the Euclidean distance metric.


This filtered set performs slightly (~1%) better than the raw dataset, but not by much. 

## Conclusion

Metric learning using training triplets does increase accuracy, but not by far. Several explanations are possible:
* There were not enough answers, i.e. training triplets to perform both the training and the testing correctly
* People were not coherent enough in their answers (we could've checked for instance that there were no contradicting answers). This has been naively taken care of in "Small improvements in metric learning", but can probably be improved by filtering further on incorrect relationship
* The "universal" music distance metric is a chimera, and people have actually really (and not slightly) different interpretation of "music distance"

However, it doesn't mean that training triplets are useless per se. This code could perfectly be reused in another field with less subjective answers (which is the "why" of this blogpost - show working code for metric learning using training triplets, as I haven't been able to find any). One could also try to derive "personalized" distance metrics, instead of trying to find a general and universal one, by asking a specific user to fill in this kind of survey on their own music, and derive a metric from their personal taste.
