---
layout: post
title:  Is trust rank a Markov chain?
date: 2020-03-07T06:00:00.000Z
author: Haoxuan (Horace)
category:
     - coding
tags:
     - 学术
comments: true
---

So literally this is a note for answering a [Zhihu question](https://www.zhihu.com/question/373709342) (the question itself is in Mandarin), but hopefully if it can help someone. The question asked why the "inital seeds" in Trust Rank could have an effect the final results if it is a Markov chain, because we know that the final convergence value of a Markov process should only depending on the transition matrix, which normally considered as the link topology in between web pages (note this "which", the question actually come from here).

Probably this is a boring topic with no-body-care models, but still we can have a quick discussion on this. Before start, I assume the reader has some basic knowledge in linear algebra, and know the basics about page rank and trust rank.

The short answer to this question: yes (if I do not make any mistakes), Trust Rank is a Markov chain, with the selected initial seed vector is a essential part to determine its transition matrix. 

I would start with page rank to have a unified math annotations and then move to trust rank very briefly. The Page Rank part comes from the [link](http://statweb.stanford.edu/~tibs/sta306bfiles/pagerank/ryan/01-24-pr.pdf). Credit to Prof. Ryan Tibshirani in Stanford.

Suppose we have $n$ websites, then what a search engine does is to sort these $n$ websites. Assuming that $p\_i$ is the score (weight) of the website, $p = \left(\begin{array} {cc} p\_1 \\ p\_2 \\ \vdots \\ p\_n \\ \end{array} \right)$. Naturally, search engines can rank the websites according to this score $p$

(to continue)

