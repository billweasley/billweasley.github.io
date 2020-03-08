---
layout: post
title:  Is Trust Rank a Markov chain?
date: 2020-03-07T06:00:00.000Z
author: Haoxuan (Horace)
category:
     - coding
tags:
     - 学术
comments: true
---

So literally this is a note for answering a [Zhihu question](https://www.zhihu.com/question/373709342) (the question itself is in Chinese), but hopefully if it can help someone. The question asked why the "inital seeds" in Trust Rank could have an effect on the final ranking results. Because we know that if it is a Markov chain, the final convergence value of a Markov process should only depending on the transition matrix, which normally considered as the link topology in between web pages (note this "which", the question actually comes from here).

Probably this is a boring topic with no-body-care models, but still we can have a quick discussion on this. Before start, I assume the reader has some basic knowledge in linear algebra, and know the basics about page rank and trust rank.

The short answer to this question: yes (if I do not make any mistakes), Trust Rank is a Markov chain, with the selected initial seed vector as a essential part to determine its transition matrix. 

I would start with page rank to have uniformed math annotations and then move to trust rank very briefly. The Page Rank part comes from the [link](http://statweb.stanford.edu/~tibs/sta306bfiles/pagerank/ryan/01-24-pr.pdf). 

*Credit to Prof. Ryan Tibshirani in Stanford*

Suppose we have $n$ websites, then what a search engine does is to sort these $n$ websites. Assuming that $p\_i$ is the score (weight) of the website, $p = \left(\begin{array}{cc} p\_1 \\\\ p\_2 \\\\ \vdots \\\\ p\_n \\\\ \end{array} \right)$. Naturally, search engines can rank the websites according to this score $p$.

## Broken Rank

A website with many incoming links is likely to be a good website (of course not absolute). So we define the link matrix as follows:

Let  $L\_{ij} = \\begin{cases}     1, \\text{there is link from website $j$ pointing to website $i$}\\\\     0, \text{otherwise} \\end{cases}$,
$ L =  \\left( \\begin{array}{cc} L\_{11} & L\_{12} &  \\dots & L\_{1n} \\\\ L\_{21} & L\_{22} &  \\dots & L\_{2n} \\\\ \\vdots \\\\ L\_{n1} & L\_{n2} &  \\dots & L\_{nn} \\\\ \\end{array} \\right)$

Let $m\_j = \\sum\_{k = 1}^j L\_{kj}$, i.e. all outbound links for site $j$, $M =  \\left(   \\begin{array}{cc} m\_1 & 0 &  \\dots & 0 \\\\ 0 & m\_2 &  \\dots & 0 \\\\ \\vdots \\\\ 0 & 0 &  \\dots & m\_n \\\\ \\end{array} \\right)$

We can think of the weight $p_i$ of a website as the sum of the weights of the contributions of other websites to this website $i$, or the probability of jumping from other websites to this website. The weight of the contribution of other sites $j$ to this site $i$ can be simply deemed as $\\frac{L\_{ij}}{m\_j} p\_j$, that is, the ratio of this $L\_{ij}$ link to all outbound links of site $j$ multiplied by the site $j$ ’s own score. So we can recursively calculate $p\_i = \\sum\_{j = 1}^{n} \\frac{L\_{ij}}{m\_j} p\_j$, and this can written in matrix form $p = LM^{-1} p$

If we let $A = LM^{-1}$, the equation above can be written as $p = Ap$. We also find $p$ as the eigenvector of matrix $A$ when the eigenvalue is $1$. This equation can of course also be treated as a Markov chain. If we define $P(\\text{go from } j \\text{ to } i) = P\_{ij}$, we have $A\_ {ij} = L\_{ij} / m\_j$ i.e. $P\_{ij} = \\begin{cases} 1 / m\_j, \\text{if site } j \text{ has a link to } i \\\\ 0, \\text{otherwise} \\end{cases}$. Then $p^{(i + 1)}$ = $Ap^{(i)}$ and $A$ is the transition matrix. **Note that here we are assuming that users click on links in a uniformly distributed way, or say, in a random way.**

As we all know (or you may not know), the linking graph of the Internet is not strongly connected (octopus model), that is, some of the websites are self-contained, i.e. they have no links to external websites; and in some cases, some websites are incoming only or outgoing only. When more the graph is not strongly connected, the solution for the eigenvector of the matrix $A$ given the eigenvalue of 1 is non-unique. (An example on the Stanford's slide... but I am not going to copy it here...)

## Page Rank
What we can do to fix the issue in our "Broken Rank" is to assume that there is a small probability that the user may not jump to a new website through the outgoing link of the current website, but enter directly from the address bar. This allows that any websites can be reached directly and hence fixes the non-strong connectivity graph problem in the Broken Rank. The improved algorithm has a famous name called Page Rank. Broken Rank's $p$ is calculated as follows: $p\_i = \\sum\_{j = 1}^{n}\\frac{L\_{ij}}{m\_j} p\_j$, and the Page Rank correction for $p$ is $p\_i = \\frac{\\left(1-d \\right)}{n} + d \\sum_{j = 1}^{n}\\frac{L\_{ij}}{m\_j} p\_j$, where $d$ is a constant less than $1$ and greater than $0$. Write it in matrix form: $p = (\\frac{1-d}{n}E + dLM^{-1}) p$, where $E$ is an all $1$ matrix of size $n \\times n$. The matrix form holds obviously. Because if we treat $p$ as a (discrete) probability distribution, then $Ep = \\left(\\begin{array}{cc} 1 \\\\ 1 \\\\ \\vdots \\\\ 1 \\end{array} \\right) \\stackrel{def}{=} I $. Here we define a $n \times 1$ all $1$ vector, $I$.
We can still think of the above equation as $p^{(i + 1)} = Ap^{(i)}$, but here the "new" $A\_{\\text{page rank}}$ is slightly changed compared with the $A\_{\\text{broken rank}}$ in the Broken Rank.

## Trust Rank
Sorry for my long rambling, and the boring copying & pasting above.  Finally we could mention the Trust Rank. Still I think those rambling is essential because it is important to unify the math symbols. Honestly, I have not read the Trust Rank paper carefully, so I would sincerely apologize for any mistakes or omissions below :)

The original papers of Trust Rank and almost all articles mentioning Trust Rank on Internet will use the following equation (in matrix form):$p = (1-d)t + dLM^{-1} p$, where $t$ is the seed vector, and other parts of the equation are same as the equation defined in Page Rank above. Here I am not going to explore how the $t$ vector comes. (well I did not take a closer look... TBH.)
A fact is that we know $t^{t} E = I^{t}$. (Because the sum of the components of the seed vector $t$ is also $1$. Here the superscript $^{t}$ means transpose, for clear.)
Hence,
$$ 
p = (1 - d) t (I^t I / n) + dLM^{-1} p \\\\ 
p =\\frac{(1 - d)}{n} t t^t E E p + dLM^{-1} p \\\\ 
p =(\\frac{(1 - d)}{n} t t^t E E  + dLM^{-1}) p \\\\  
p =((1 - d) t t^t E + dLM^{-1}) p 
$$ 
Obviously for Trust Rank, we can also write the equation as $p^{(i + 1)}= A p^{(i)}$ ，and the value of $A$ needs to be determined by $t$.