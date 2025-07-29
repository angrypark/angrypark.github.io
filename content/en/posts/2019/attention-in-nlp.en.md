+++
title = "Attention in NLP"
description = "This post summarizes what attention is, focusing on several important papers and how it is used in NLP."
date = 2019-01-26T00:00:00+09:00
draft = false
tags = ["attention", "nlp", "transformer", "neural-network"]
categories = ["natural-language-processing"]
language = "en"
url = "/en/posts/2019/attention-in-nlp/"
+++

This post summarizes what attention is, focusing on several important papers and how it is used in NLP.

> **Table of Contents**
- [Problems with Existing Encoder-Decoder Architecture](#problems-with-existing-encoder-decoder-architecture)
- [Basic Idea](#basic-idea)
- [Attention Score Functions](#attention-score-functions)
- [What Do We Attend To?](#what-do-we-attend-to)
- [Multi-headed Attention](#multi-headed-attention)
- [Transformer](#transformer)

## Problems with Existing Encoder-Decoder Architecture

The most important part of the Encoder-Decoder architecture is how to vectorize the input sequence. In NLP, input sequences often have dynamic structures, so problems arise when converting them to fixed-length vectors. For example, sentences like "Hello" and "The weather is nice today but the fine dust is severe, so make sure to wear a mask when you go out!" contain very different amounts of information, yet the encoder-decoder structure must convert them to vectors of the same length. Attention was first proposed to reduce information loss and solve problems more intuitively by reflecting which parts should be paid particular attention to in sequence data, as the word suggests.

## Basic Idea (Bahdanau Attention)

> Paper: [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)

The most basic idea is to encode each word as a vector, combine them with weighted sum according to attention weights, and then use this to select what the next word should be.

The paper used this approach for NMT, using bidirectional RNN as encoder, and for the $i$th word, combining all encoder outputs to create a context vector. However, instead of simple sum, it multiplies by weight $\alpha_{ij}$ for weighted sum (first equation below). The attention weight for the $j$th word with respect to the $i$th word is created by feeding the $i$th word and $j$th original encoder output through a feedforward neural network (the model that creates attention weights is called the align model in the paper) as shown in the second equation below.

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}
$$

$$
e_{ij} = a(s_{i-1}, h_j)
$$

The align model was made as a Multi-layer Perceptron to reflect nonlinearity, and ultimately this align model exists to properly align and pair words with the same meaning in NMT (hence align). The NMT cost function itself was used as loss for backpropagation.

## Attention Score Functions

After the above paper, several variations on how to create this attention score $\alpha_{ij}$ emerged. Let me summarize them. To unify terminology, let's call the decoder state we want to create as $q$ (query vector), and all encoder states used here as $k$ (key vector) (this definition comes from the Attention is All You Need paper that we'll cover later). Using these terms, $\alpha_{ij}$ can be considered as the attention score between the $i$th query vector and $i-j$ key vectors.

### (1) Multi-layer Perceptron (Bahdanau et al. 2015)

$$
a(q, k) = w_2^T \tanh (W_1[q;k])
$$

This is the MLP from the above paper rewritten. This method has the advantage of being flexible and good for large datasets.

### (2) Bilinear (Luong et al. 2015)

$$
a(q, k) = q^TWk
$$

The Luong Attention from the same year multiplies $q$ and $k$ with one weight matrix $W$ to create it.

### (3) Dot Product (Luong et al. 2015)

$$
a(q, k) = q^Tk
$$

Similar to 2, but simply takes the dot product of $q$ and $k$ to use as attention. This is good because there are no parameters to learn, but the disadvantage is that $q$ and $k$ must have the same length.

### (4) Scaled Dot Product (Vaswani et al. 2017)

$$
a(q, k) = \frac{q^Tk}{\sqrt{\mid{k}\mid}}
$$

This is a paper that improved on 3. The basic approach is that the dot product result increases proportionally to the dimensions of $q$ and $k$, so we divide by the vector size.

## What Do We Attend To?

All the methodologies so far have used attention on RNN outputs of input sentences and utilized them for decoding. Now let's look at various ways to apply attention.

### (1) Input Sentence

The most basic method is to give attention to input sentences before/after.

#### - Copying Mechanism (Gu et al. 2016)

> Paper: [Incorporating Copying Mechanism in Sequence-to-Sequence Learning](https://arxiv.org/pdf/1603.06393)

This method was first proposed when words from input sequences frequently overlap in output sequences, to copy them well. For example, when leading a conversation, you often need to use words that have appeared before to answer.

*This post contains only part of the original. You can check the full content on the previous blog.* 