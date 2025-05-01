# Assignment 9: Transformer-Based Sentiment Analysis on IMDB

This repository contains three Colab notebooks exploring different approaches to sentiment analysis using transformer architectures and NLP tools in TensorFlow, Keras, and KerasNLP. The project covers the full spectrum: using a pretrained model for quick inference, fine-tuning a backbone for improved accuracy, and building a transformer from scratch for deep understanding.

---

## Table of Contents

- [Overview](#overview)
- [Part 1: Inference with a Pretrained Classifier](#part-1-inference-with-a-pretrained-classifier)
- [Part 2: Fine-tuning a Pretrained Backbone with Keras NLP](#part-2-fine-tuning-a-pretrained-backbone-with-keras-nlp)
- [Part 3: Building and Training a Transformer from Scratch](#part-3-building-and-training-a-transformer-from-scratch)
- [YouTube Playlist](#youtube-playlist)
- [Next Steps](#next-steps)
- [Author](#author)

---

## Overview

The IMDB dataset is a widely-used benchmark for binary sentiment classification. This assignment demonstrates a progression of modeling techniques:

1. Use of a **fully pretrained model** for zero-shot inference,
2. **Fine-tuning a pretrained transformer backbone** using `keras-nlp`,
3. **Constructing a transformer model from scratch**, including attention, positional encoding, and encoder blocks.

Each notebook builds on the previous one in terms of customization and learning depth.

---

## Part 1: Inference with a Pretrained Classifier

**Notebook:** [Open in Colab](https://colab.research.google.com/drive/1r0iPvonevckTjL3dBJpKRaQaS194LjnN?usp=sharing)

This notebook demonstrates how to use a **pretrained BERT classifier** from `keras-nlp` for performing sentiment analysis without any training or fine-tuning.

**Highlights:**
- Uses `keras_nlp.models.BertClassifier`.
- Loads model and vocabulary from TensorFlow Hub.
- Tokenizes input text with the correct preprocessing pipeline.
- Runs predictions on user-defined examples.
- Designed for minimal setup and fast deployment.

This is ideal for benchmarking or quickly integrating NLP into applications.

---

## Part 2: Fine-tuning a Pretrained Backbone with Keras NLP

**Notebook:** [Open in Colab](https://colab.research.google.com/drive/1jcVVO27-cOZcKPZ-rTUnNBsN2K73vFTB?usp=sharing)

This notebook shows how to fine-tune a **pretrained transformer encoder**, such as BERT, on the IMDB dataset using the KerasNLP API.

**Highlights:**
- Loads a BERT backbone via `keras_nlp.models.BertBackbone`.
- Adds a custom classification head (GlobalPooling + Dense).
- Fine-tunes on tokenized IMDB movie reviews.
- Tracks accuracy and loss across training and validation sets.
- Uses early stopping and learning rate scheduling callbacks.

Fine-tuning gives better performance than zero-shot inference because it adapts model weights to the domain of the task.

---

## Part 3: Building and Training a Transformer from Scratch

**Notebook:** [Open in Colab](https://colab.research.google.com/drive/1OSK_J5yyTsryyC5bPg_KopgwCC6cxzfK?usp=sharing)

This is the most technically involved notebook. It builds a transformer-based classifier **entirely from scratch** using core Keras layers, providing full transparency into how the architecture works internally.

**Highlights:**
- Manually implements:
  - Positional Encoding
  - Multi-Head Attention
  - Feedforward sublayers
  - Layer Normalization
  - Transformer Encoder Blocks
- Combines these into a custom text classification model.
- Trains on padded IMDB sequences using the Keras `Model` API.
- Includes:
  - Custom attention mechanism
  - GlobalAveragePooling over encoder output
  - Visualization of predictions
- Includes scaffold for attention weight visualization (future work).

This is ideal for anyone learning how transformer internals are constructed and trained.

---

## YouTube Playlist

Watch the full video walkthrough for all three notebooks, including explanations and debug traces:

**Playlist:** [Transformer Sentiment Analysis on IMDB - YouTube](https://www.youtube.com/playlist?list=YOUR_PLAYLIST_ID_HERE)

---

## Next Steps

Ways to build on this project:

- Add attention visualization for interpretability.
- Compare performance across notebooks using consistent metrics.
- Try different transformer sizes or architectures (e.g., GPT-style).
- Switch to Hugging Face Transformers for broader model access.
- Apply transfer learning to other datasets beyond IMDB.
- Convert notebooks to scripts and deploy with Flask or FastAPI.

---


## Author

This project was developed as part of **Assignment 9: Transformers and NLP**, the final project in a graduate-level course on deep learning and natural language processing.
