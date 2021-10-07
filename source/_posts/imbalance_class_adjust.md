---
layout: post
title:  Handle imbalance class by adjusting output distribution
date: 2021-10-06T11:00:00.000Z
author: Haoxuan (Horace)
category:
     - coding
tags:
     - 学术
comments: true
---

## Some nonsonses

It becomes somehow boring in my current work - well, what one should expect from a bank's job? Those trivial data pre-processing/ETL actually made my days, and these thing really push me back from login my working laptop every morning. There are no interesting projects here, because of the poor management and poor understanding of data science from management team. There is no labeled data and computational resources here at all. Even for unlabeled data, the comapany does not provide any really useful meta-data or data dictionary. It is hell for any data scientists I believe - one can only guess everything unless you find the correct person who understands the business/dataset - this often takes days and even months. 

Despite all these frustrations I came across in my early career, there are still some flashes there. For example, we do have a share session - a few weeks ago, one of my colleagues shared the commonly used methods for handling label-imbalanced datasets - basically a summary of those general methods like down/up-sampling, SMOTE, etc. His presentation was discussed warmly among us because it is always a normal circumstance for those who have been working in the industry for a while to find that they need to handle imbalanced datasets unless they produce some dataset themselves. Unfortunately, the real world is very ugly - in general, the human world follows a power-law distribution, where you will often see the majority and the minority class - or a long-tail distribution rather than a sort of uniform distribution/gaussian distribution. Prof. Adam Kelleher gave us his suggestion dusing the duscussion - I do hope I correctly understand his idea and try to note it here becasuse I think this is something useful. I am not from a math background - in fact, I have failed my exames for a few times during my studies when it comes to math...so please do not hestiate figure out any mistakes there if you could identify one.

Here we go.

## Main things

Up-sampling/down-sampling/reweighting are common ways of making a data become balanced - well this essentially changes the distribution of data and may imported some baises there. We now consider a slightly change that does not affect this distribution too much in a naive Bayes classifier setting. The reason we consider a naive Bayes classifier is that its clearly build the mapping between the prior and the posterior and hence we can easily see how our adjustments to the data affect the distribution.

Given a set of labeled data $T$ with data $\mathcal{X}$ and labels $\mathcal{Y}$, a corresponding $\mathcal{Y}$ adjusted data $A$, and a naive Bayes classifier fitted on $A$, $P\_A(\mathcal{Y}=y | \mathcal{X})$. We want to know $P\_T(\mathcal{Y}=y|\mathcal{X})$. Assume $A$ has only slight changes from $T$. i.e. $P_A(\mathcal{X} = x) \approx P_T (\mathcal{X} = x), \forall x \in \mathcal{X}$ .

Simply as in bayes rule:
{% raw %}
$$
P_A(\mathcal{Y} = y | \mathcal{X})  = \frac{P(\mathcal{X}  | \mathcal{Y} = y) P_A(\mathcal{Y}=y)}{P_A(\mathcal{X})}
$$
{% endraw %}
{% raw %}
$$
P_T(\mathcal{Y} = y | \mathcal{X})  = \frac{P(\mathcal{X}  | \mathcal{Y} = y) P_T(\mathcal{Y}=y)}{P_T(\mathcal{X})}
$$
{% endraw %}

In our settings, we only adjusted $P_T(\mathcal{Y})$ to $P_A(\mathcal{Y})$. Simply by a ration, we have
{% raw %}
$$
\frac{P_A(\mathcal{Y} = y | \mathcal{X})}{P_T(\mathcal{Y} = y | \mathcal{X})} = \frac{P_A(\mathcal{Y})}{P_T(\mathcal{Y})}
$$
{% endraw %}
Hence,
{% raw %}
$$
P_T(\mathcal{Y} = y | \mathcal{X})  = \frac{P_A(\mathcal{Y} = y | \mathcal{X}) P_T(\mathcal{Y})}{P_A(\mathcal{Y})}

$$
{% endraw %}
The equation above can be generalised as long as $P_A(\mathcal{X})$ does not changed much from $P_T(\mathcal{X})$ .
