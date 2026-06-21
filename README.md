# Straining the Great Southern Melting Pot

## 📋 Overview

*This README provides an overview of the repository structure. Since a report was required for this assignment, the README serves as a brief guide to explain the purpose and contents of each folder.*

The project applies text mining techniques to a slice of 2023 Atlanta restaurant reviews, covering data understanding, general data preparation, multilabel classification, sentiment analysis, and named entity recognition.

## 📁 Repository Structure

```tree
text_mining_in_reviews/
│
├── data/
│   ├── 00_atlanta_restaurant_slice_2023.csv
│   ├── 01_atlanta_restaurant_slice_2023_new_features.csv
│   ├── 02_atlanta_restaurant_slice_2023_translated.csv
│   └── 03_atlanta_restaurant_slice_2023_translated_corrected_tokens.csv
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_general_data_prep.ipynb
│   ├── 03_multilabel_classification.ipynb
│   ├── 04_sentiment_analysis.ipynb
│   ├── 05_named_entity_recognition.ipynb
│   ├── cuisine_network.html
│   └── lib/
│
├── source/
│   ├── evaluation.py
│   ├── general_preprocessing.py
│   ├── modelling.py
│   ├── my_utils.py
│   ├── ner_graph_prep.py
│   ├── sentiment_prep.py
│   └── visualizations.py
│
├── README.md
├── group08_report.pdf
└── requirements.txt
```

## 👥 Team

- Beatriz Marques – 20231605
- David Carrilho – 20231693
- Duarte Fernandes – 20231619
- Filipe Caçador – 20231707
- Mariana Calais-Pedro – 20231641