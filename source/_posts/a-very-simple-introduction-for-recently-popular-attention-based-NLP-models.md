---
layout: post
title:  A very simple introduction for recently popular attention based NLP models
date: 2019-07-22T11:00:00.000Z
author: Haoxuan (Horace)
category:
     - coding
tags:
     - 学术
comments: true
bibliography:
     - 'a-simple-and-concise-introduction-for-recent-popular-attention-based-NLP-model.bib'
---

This is a draft for literature review part of my master thesis. Sorry if any mistakes there and kindly please let me know if you found some mistakes.

I will briefly present some background concepts literature review for the state-of-the-art attention based deep learning models (BERT and XLNet) that is planned to be used in my thesis project.

## Attention

The recurrent neural network (RNN) was a major neural network architecture that used in NLP field. RNN has its drawbacks for the inability for capturing long inputted features (called “gradient vanishing” [4]). Some RNN variant using “gate method” (e.g. LSTM[5], GRU [1]) to mitigate the problem but cannot resolve this. A innovative way to solve this is to using “attention” mechanism [8]. Given a sequence of source tokens $S$ of length $T$, where $\bar{h}\_i$ is the hidden representation for a source token the position $i \in [1, T]$ in a RNN. The context vector $c\_t$ for a token in the target position $t$ can be calculated as: {% raw %}$$c_t = \sum_{i =1}^{T} \alpha_{t, i}\bar{h}_i$${% endraw %} where,{% raw %}$$\alpha_{t,i}=\frac{exp(score(h_t,\bar{h}_i))}{\sum^{T}_{i^{'}= 1}exp(score(h_t, \bar{h}_{i^{'}}))}$${% endraw %} $h\_t$ is the hidden representation for the target token in the position $t$. A score function here could be a dot product of $h\_t$ and $\bar{h}\_i$, it also could be ${\bar{h}\_i}^{\mathsf{T}}W\_{\alpha}h\_t$, ${v\_{\alpha}^{\mathsf{T}}W\_{\alpha}[h\_t;\bar{h}\_i]}$ or a neural network approximation (here $v\_{\alpha}$ and $W\_{\alpha}$ are trainable parameters).

## Self-attention [8, 11]

Attention mechanism could give a better encoder summary representation, say $s$, than that produced from the vanilla RNN, say $h\_T$ (with input sequence $(x\_1, ... x\_T)$ and corresponding hidden states $(h\_1, ... h\_T)$ of the vanilla RNN): \begin{aligned} u\_t &= \tanh(Wh\_t) \\\\ \alpha\_t &= \frac{exp(score(u\_t, u))}{\sum^{T}\_{t=1}exp(score(u\_t, u))} \\\\ &= \sum^{T}\_{t=1} \alpha\_t h\_t\end{aligned}  where $u$ could be randomly initialised.

## Transformer [9]

Transformer is a encoder-decoder architecture that completely drops traditional sequential way. It presents a more complicated attention mechanism. In this mechanism, each input token associates with a key vector $k$ of dimension $d\_k$, a query vector $q$ of dimension $d\_k$ and a value vector $v$ of dimension $d\_v$. The token on index $i$ in an input of length $n$, its attention value between a token on index $j$ is:
{% raw %}$${attention(q,k,v)}_{i, j} = \frac{exp(q_i^{\mathsf{T}}k_j)}{\sqrt{d_k}\sum_{j^{'} = 1}^{n} exp(q_i^{\mathsf{T}}k_{j^{'}})}v_j$${% endraw %}
In matrix form:
{% raw %}$$Attention(Q, K, V) = Softmax(\frac{QK^{\mathsf{T}}}{\sqrt{d_k}})V$${% endraw %}
The attention mechanism is a sublayer in the network and a full layer is `LayerNorm(x + Sublayer(x))`. An transformer encoder uses the full attention layer result and pass the result to a feed-forward layer with residual connection and normalisation.
The decoder of the a transformer still uses sequential input. It uses decoder inputs as queries, and the encoder output value as keys and values. The first sublayer of the decoder is “masked” because it does not process the attention for those tokens after the current input token (which has not been inputted yet). Another innovative point of the model is the following. The input embedding for both encoder and decoder contains a positional encoding to compensate the information might be lost compared with traditionally sequential models.

## 2-stage multi-task transfer learning [6,7]

The paper [6] proposed a revolutionary framework, using a large set unsupervised corpus to do a pre-training and a small set of task-specified supervised dataset to do a fine turning. This reduces a lot on the amount of labelled data, which could be very cost. The paper uses a Transformer decoder to do the unsupervised pre-training called generative pre-training transformer (GPT). Specifically, the model maximise following likelihood during pre-training stage: {% raw %}$$L_1(\mathcal{U}) = \sum_i \log P(u_i|u_{i-k}, ... , u_{i-1};\Theta)$${% endraw %}
where $\mathcal{U} = \{u\_1, ..., u\_n\}$ is an unsupervised carpus of tokens and k is a hyper-parameter for context token window. The probability distribution over tokens, $P(\cdot)$, produced by the Transformer decoder. After finishing pre-training, the model then will can be adjusted for task-specific supervised learning using few labelled data. The object is to maximise the following: {% raw %}$$L_2(\mathcal{C}) = \sum_{(x, y)} \log P(y|x^{1}, ... , x^{m};\Theta)$${% endraw %} where $x^{1}, ... , x^{m}$ are a sequence of tokens of length $m$ and $y$ is the corresponding label. $P(\cdot)$ here is produced by a linear output layer to predict $y$. A following work [7], GPT-2, shows a larger unsupervised data set can effectively enhance the performance of this model on many tasks.

## BERT [3]

Bidirectional Encoder Representations from Transformers (short for “BERT”) is another 2-stage multi-task transfer learning model using Transformer encoder in the pre-training stage. This makes it becomes it performs better than GPT [6]in most tasks by using bidirectional attention. During pre-training stage, instead predicting next token method that used in GPT, BERT has 2 sub-tasks: Masked LM, which is to predict randomly masked tokens (i.e. tokes that are replaced by a special “\[MASK\]” token), and Next Sentence Prediction, which is to predict whether a following sentence is actual next sentence in the context or a randomly selected sentence from corpus.

## GPT and BERT: A different perspective

The paper [10] proposed a different perspective to view GPT [6,7] and BERT [3], autoregressive language model and autoencoding language model.

 Autoregressive language model estimates the probability distribution of a text sequence, say $x = (x\_1, .., x\_T)$ by factorising the its likelihood into $p(\mathbf{x}) = \prod\_{t = 1}^{T} p(x\_T|\mathbf{x}\_{\lt t})$ or $p(\mathbf{x}) = \prod\_{t = T}^{1} p(x\_T|\mathbf{x}\_{\gt t})$. Specifically, a forward AR maximise the following likelihood: \begin{aligned}\mathop{max}\_\mathbf{\theta} \log p\_{\theta}(\textbf{x})  &=  \mathop{\sum}\_{t=1}^{T}\log p\_\theta(x\_t | \mathbf{x}\_{\lt t}) \\\\ &= \mathop{\sum}\_{t=1}^{T} \log \frac{\exp(h\_\theta (x\_{1:t-1}^{T}e(x\_t))}{\sum\_{x^{'}} \exp(h\_\theta (x\_{1:t-1}^{T})e(x^{'}))}\end{aligned} where $h\_\theta(x\_{1:t-1})$ is a context representation produced by neural network (e.g. a uni-directly masked transformer or a RNN) and $e(x)$ is the embedding of $x$. Autoregressive model uses uni-directional language encoding and hence cannot form an effective pre-training objective function for some natural language understanding (NLU) tasks that need bidirectional context information.

Autoencoding language model reconstructs the original language from its noisy corrupted version. It attempts to maximise the following likelihood: \begin{aligned} \mathop{max}\_\mathbf{\theta} \log p\_{\theta}(\bar{\textbf{x}} | \hat{\textbf{x}})  & \approx \mathop{\sum}\_{t=1}^{T}m\_t\log  p\_\theta(x\_t | \hat{\mathbf{x}}) \\\\ &= \mathop{\sum}\_{t=1}^{T} m\_t \log \frac{\exp(H\_\theta {(\hat{\textbf{x}})}\_{t}^\top e(x\_t))}{\sum\_{x^{'}} \exp(H\_\theta {(\hat{\textbf{x}})}\_{t}^\top e(x^{'}))}\end{aligned} where $\bar{\textbf{x}}$ indicates the set of corrupted (i.e. masked) tokens. $\hat{\textbf{x}}$ is the corrupted version of $\textbf{x}$. $H\_\theta$ is a Transformer mapping a text sequence $\textbf{x}$ of length $T$ to a sequence of vector representation, $H\_\theta(\textbf{x})=[{H\_\theta(\textbf{x})}\_1,{H\_\theta(\textbf{x})}\_2,...,{H\_\theta(\textbf{x})}\_T]$. $m\_t$ indicates whether the token at position $t$ is a noisy version of original token. Autoencoding model accesses the 2 side context information so it overcomes the weakness of the autoregressive models, however it assumes the conditional independence when factoring likelihood (hence there is a $\approx$ in the equation above) and it may import some noise token in the noisy version of text (e.g. ’\[mask\]’ token in BERT [3]), which do not appear in the fine turning stage.

## XLNet [10]

The paper then propose a new “permutation” language model to utilise the advantages in both autoregressive and autoencoding model. Specifically,it maximise the following objective:
{% raw %}$$\mathop{max}_\mathbf{\theta} \mathbb{E}_{\textbf{z} \sim \mathcal{Z_T}} \left[\mathop{\sum}_{t=1}^{T}\log p_\theta(x_{z_t} | \mathbf{x}_{z_{\lt t}})\right]$${% endraw %}
where $\mathcal{Z\_T}$ is the set of all permutations for index sequence $[1, 2, ... T]$, where includes $T!$ unique element. $z\_t$ and $\textbf{Z}\_{\lt t}$ is the $t$-th element and the first $t-1$ elements of a permutation $\textbf{z} \in \mathcal{Z\_T}$. The factorisation order of the likelihood is according to the sampled permutation order and hence it is capture bidirectional context. The objective is in Autoregressive form and hence avoids the independence assumption and the difference in between pre-training and fine-turning. The input is still in sequence order with positional encodings and factorisation permutation is achieved by the masks used in Transformers. A standard softmax parameterisation under the new objective will lead a positional-irrelevant trivial result (i.e. the prediction result will be independent with the value of the target token position $z\_t$). To resolve this problem, the paper gives a new type of representations, $g\_\theta(\textbf{x}\_{\textbf{z}\_{\lt t}},z\_t)$, which takes $z\_t$ as an additional input.
$$p\_\theta(\textbf{X}\_{z\_t}=x|\textbf{x}\_{\textbf{z}\_{\lt t}})
     = \frac{\exp(e(x)^\top g\_\theta (\textbf{x}\_{\textbf{z}\_{\lt t}}, z\_t))}{\sum\_{x^{'}} \exp(e(x^{'})^\top g\_\theta (\textbf{x}\_{\textbf{z}\_{\lt t}}, z\_t))}$$
There is still a contradiction for formulating $g\_\theta (\textbf{x}\_{\textbf{z}\_{t}}, z\_t)$. To predict $x\_{z\_t}$, $g\_\theta (\textbf{x}\_{\textbf{z}\_{\lt t}}, z\_t)$ should not contains the $x\_{z\_t}$ to avoid trivial results; however, to predict $x\_{z\_j}, \forall j \gt t$, $g\_\theta (\textbf{x}\_{\textbf{z}\_{\lt t}}, z\_t)$ should uses the content $x\_{z\_j}$. The author uses a trick called Two Stream Self-attention to resolve the problem. Specifically, it contains 2 sets of hidden representations, the content representation $h\_\theta(\textbf{x}\_{\textbf{z}\_{\leq t}})$ (or say $h\_{z\_t}$) and the query representation $g\_\theta (\textbf{x}\_{\textbf{z}\_{\lt t}}, z\_t)$ (or say $g\_{z\_t}$), rather than traditionally single set representations. For self-attention layer $ m \in [1, ..., M]$, the query stream $g\_{z\_t}^{(m)}$ updates generally (i.e. by omitting details like multi-head attention, residual connection, layer normalisation and some other details) with: {% raw %}$$g_{z_t}^{(m)} \longleftarrow Attention(\textbf{Q} = g_{z_t}^{(m - 1)}, \textbf{KV} = \textbf{h}_{\textbf{Z} \lt t}^{(m - 1)};\theta)$${% endraw %} and the content stream updates with: {% raw %}$$f_{z_t}^{(m)} \longleftarrow Attention(\textbf{Q} = f_{z_t}^{(m - 1)}, \textbf{KV} = \textbf{h}_{\textbf{Z}\leqslant t}^{(m - 1)};\theta)$${% endraw %}
Where Q,K,V devote query, key and value in an attention operation. $Attention(\cdot)$ is a standard self-attention and the 2 attention stream share a same set of parameters. For initialisation, $g\_i^{(0)}$ is initialised with a trainable vector and $h\_i^{(0)}$ is the corresponding word embedding for input position $i$.
To reduce the challenges in optimisation raised from permutation, another trick the author proposed is only to predict the last few tokens given a factorisation order. Given a cutting point parameter $c$,
{% raw %}$$\mathop{max}_{\mathbf{\theta}} \mathbb{E}_{\textbf{z} \sim \mathcal{Z_T}} \left[\log p_\theta(\textbf{x}_{\textbf{z} \gt c } | \textbf{x}_{\textbf{z}_{\leqslant c}})\right] = \mathbb{E}_{\textbf{z} \sim \mathcal{Z_T}} \left[{\mathop{\sum}}_{t= c + 1}^{|\textbf{z}|} \log p_\theta(x_{z_t} | {\textbf{x}}_{\textbf{z}_{\leqslant c}})\right]$${% endraw %} XLNet also integrates some techniques from Transformer-XL [2] for enhancing its capability for long text. Specifically, it uses recurrence mechanism, which integrates the content representation of last segment when updating the hidden attention representations of tokens in current segment, and relative positional encodings, a positional encoding way considering whether two tokens in a same segment.

<hr>
# References
[1]  Kyunghyun   Cho,   Bart   van   Merrienboer,   C ̧ aglar   G ̈ul ̧cehre,   FethiBougares, Holger Schwenk, and Yoshua Bengio.  Learning phrase repre-sentations using RNN encoder-decoder for statistical machine transla-tion.CoRR, abs/1406.1078, 2014.

[2]  Zihang Dai, Zhilin Yang, Yiming Yang, Jaime G. Carbonell, Quoc V. Le,and Ruslan Salakhutdinov.  Transformer-xl:  Attentive language modelsbeyond a fixed-length context.CoRR, abs/1901.02860, 2019.

[3]  Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova.BERT: pre-training of deep bidirectional transformers for language un-derstanding.CoRR, abs/1810.04805, 2018.

[4]  S.  Hochreiter,  Y.  Bengio,  P.  Frasconi,  and  J.  Schmidhuber.   Gradientflow in recurrent nets:  the difficulty of learning long-term dependencies.In S. C. Kremer and J. F. Kolen, editors,A Field Guide to DynamicalRecurrent Neural Networks. IEEE Press, 2001.

[5]  Sepp  Hochreiter  and  J ̈urgen  Schmidhuber.   Long  short-term  memory.Neural Comput., 9(8):1735–1780, November 1997.

[6]  Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever.Improving language understanding by generative pre-training.  2018.

[7]  Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, andIlya Sutskever.  Language models are unsupervised multitask learners.2019.

[8]  Antoine  J.-P.  Tixier.Notes  on  deep  learning  for  NLP.CoRR,abs/1808.09772, 2018.

[9]  Ashish  Vaswani,  Noam  Shazeer,  Niki  Parmar,  Jakob  Uszkoreit,  LlionJones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attentionis all you need.CoRR, abs/1706.03762, 2017.

[10]  Zhilin  Yang,  Zihang  Dai,  Yiming  Yang,  Jaime  G.  Carbonell,  RuslanSalakhutdinov, and Quoc V. Le. Xlnet:  Generalized autoregressive pre-training for language understanding.CoRR, abs/1906.08237, 2019.

[11]  Zichao Yang, Diyi Yang, Chris Dyer, Xiaodong He, Alexander J. Smola,and  Eduard  H.  Hovy.   Hierarchical  attention  networks  for  document classification.  InHLT-NAACL, 2016.
