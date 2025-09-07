# AI-ML-internship-task4
---

📰 News Topic Classifier Using BERT

1. Objective

The objective of this project is to build a news topic classifier that automatically categorizes news headlines into predefined categories: World, Sports, Business, and Sci/Tech. The aim is to fine-tune a pre-trained transformer model (BERT) on the AG News dataset, evaluate its performance using standard metrics, and deploy it with an interactive web interface for real-time predictions.


---

2. Methodology / Approach

a) Dataset

Dataset Used: AG News Dataset

Description: 120,000 training samples and 7,600 test samples, each labeled into one of four categories (World, Sports, Business, Sci/Tech).

Source: Hugging Face Datasets (load_dataset("ag_news")).


b) Preprocessing

Tokenization performed using BERT Tokenizer (bert-base-uncased).

Headlines were truncated/padded to a maximum length of 128 tokens.

Labels were mapped to integers for training.


c) Model Architecture

Pre-trained BERT base (uncased) model with a classification head.

Number of output labels = 4.

Fine-tuning performed using Hugging Face Trainer API.


d) Training Setup

Learning rate: 2e-5

Batch size: 16

Epochs: 2–3

Optimizer: AdamW

Loss Function: Cross-Entropy Loss

Evaluation Metrics: Accuracy and Weighted F1-score


e) Evaluation

The trained model was evaluated on the test dataset.

Performance was measured using Accuracy and F1-score.


f) Deployment

The fine-tuned model was deployed using Gradio.

Users can enter a news headline, and the model returns predicted topic probabilities.



---

3. Key Results

Accuracy: ~94–95%

F1-score (Weighted): ~94–95%


Observations

BERT successfully learned the semantic meaning of headlines.

Misclassifications mostly occurred between Business and Sci/Tech due to overlapping content.

The model showed strong generalization capabilities even with limited training epochs, thanks to transfer learning.


Example Prediction

Input: “Microsoft releases new AI-powered search engine”

Output: Sci/Tech
