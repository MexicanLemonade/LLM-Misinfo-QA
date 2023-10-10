# LLM-Misinfo-QA

**Under construction**

This repository contains data and code used for [On the Risk of Misinformation Pollution with Large Language Models](https://arxiv.org/abs/2305.13661) (to appear on Findings of EMNLP 2023). 

## Data
NQ-1500.tsv and CovidNews.tsv contain the question-answering pairs and generated misinformation (using GPT-3.5) in the experiments.

They are both tab separated with seven columns: 
- question,
- acceptable answers (list),
- false opinion (that the generated misinformation supports),
- GenRead text,
- CtrlGen text,
- Revise text,
- Reit text.

Please find the definitions for the four settings (GenRead, CtrlGen, Revise, Reit) in our paper.

## Code
