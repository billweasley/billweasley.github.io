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

So literally this is a note for answering a [Zhihu question](https://www.zhihu.com/question/373709342) (the question itself is in Mandarin), but hopefully if it can help someone. The question asked why the "inital seeds" in Trust Rank could have an effect the final ranking results. Because we know that if it is a Markov chain, the final convergence value of a Markov process should only depending on the transition matrix
which normally considered as the link topology in between web pages (note this "which", the question actually comes from here).

Probably this is a boring topic with no-body-care models, but still we can have a quick discussion on this. Before start, I assume the reader has some basic knowledge in linear algebra, and know the basics about page rank and trust rank.

The short answer to this question: yes (if I do not make any mistakes), Trust Rank is a Markov chain, with the selected initial seed vector is a essential part to determine its transition matrix. 

I would start with page rank to have uniformed math annotations and then move to trust rank very briefly. The Page Rank part comes from the [link](http://statweb.stanford.edu/~tibs/sta306bfiles/pagerank/ryan/01-24-pr.pdf). Credit to Prof. Ryan Tibshirani in Stanford.

Suppose we have $n$ websites, then what a search engine does is to sort these $n$ websites. Assuming that $p\_i$ is the score (weight) of the website, $p = \left(\begin{array}{cc} p\_1 \\\\ p\_2 \\\\ \vdots \\\\ p\_n \\\\ \end{array} \right)$. Naturally, search engines can rank the websites according to this score $p$.

## Broken Rank

A website with many incoming links is likely to be a good website (of course not absolute). So we define the link matrix as follows:

Let  $L\_{ij} = \\begin{cases}     1, \\text{there is link from website $j$ pointing to website $i$}\\\\     0, \text{otherwise} \\end{cases}$,
$ L =  \\left( \\begin{array}{cc} L\_{11} & L\_{12} &  \\dots & L\_{1n} \\\\ L\_{21} & L\_{22} &  \\dots & L\_{2n} \\\\ \\vdots \\\\ L\_{n1} & L\_{n2} &  \\dots & L\_{nn} \\\\ \\end{array} \\right)$

Let $m\_j = \\sum\_{k = 1}^j L\_{kj}$, i.e. all outbound links for site $j$, $M =  \\left(   \\begin{array}{cc} m\_1 & 0 &  \\dots & 0 \\\\ 0 & m\_2 &  \\dots & 0 \\\\ \\vdots \\\\ 0 & 0 &  \\dots & m\_n \\\\ \\end{array} \\right)$