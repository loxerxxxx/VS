A Desperate Psychopath’s Idea Machine
=====================================

Exploring the long tail of startup and go-to-market ideas using LLMs and Verbalized Sampling.

## Overview

This project is a Streamlit research demo for generating startup and GTM ideas and comparing prompting strategies:

- Direct Prompting
- Verbalized Sampling

It supports both quick brainstorming and controlled experiments focused on diversity and mode collapse.

## Key Insight

Diverse idea generation can be evaluated, not just observed. By measuring embedding similarity and long-tail coverage, the app surfaces whether a prompting strategy is collapsing into repetitive ideas.

## Method

1. Generate ideas for a given topic with one or two prompting strategies.
2. Embed ideas using `sentence-transformers` (`all-MiniLM-L6-v2`).
3. Compute mode collapse and long-tail metrics.
4. Visualize idea geometry using PCA and interactive plots.

## Mode Collapse Score

Mode Collapse Score is the average pairwise cosine similarity among idea embeddings.

- Higher score -> stronger collapse (ideas are more similar)
- Lower score -> better diversity

In Experiment Mode, scores are computed for both Direct Prompting and Verbalized Sampling and compared in a bar chart.

## Long Tail Discovery Score

Long Tail Discovery Score is:

`number_of_clusters / total_ideas`

It provides a coarse estimate of how widely generated ideas span conceptual space.

## Experiment Reproducibility

Run locally:

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

Optional analysis script:

```bash
python3 scripts/mode_collapse.py
```
