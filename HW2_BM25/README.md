# Homework 2 - BM25

## 使用的 tool
Python, nltk, tqdm, re

## 資料前處理

使用stopword, lemmatization，以及正規表達式，將一些特殊符號或數字刪除。

## 模型參數調整

* BM25 term weight 公式：

$$sim_{BM25}\left (d_{j}, q \right ) \equiv \sum_{w_{j}\in \left ( d_{j} \cap q \right )}^{} IDF(w_{j}) \times \frac{ \left ( K_{1} + 1 \right ) \times tf_{i,j}}{K_{1} [\left ( 1 - b \right ) + b \times \frac{len\left ( d_{j} \right )}{avg_{doclen}}] + tf_{i,j}} \times \frac{\left ( K_{3} + 1 \right ) \times tf_{i,q}}{ K_{3} + tf_{i,q}}$$

* IDF 公式：

$$IDF(w_{j}) = log\left ( \frac{N - n_{i} + 0.5}{n_{i} + 0.5} \right )$$

* 最終使用參數（Kaggle Public Score: 0.71854）：
    * `K1` = `1.8`
    * `K3` = `1.8`
    * `b` = `0.75`


## 模型運作原理

BM25 跟 VSM 最大的差異在於多處理了 Document Length Normalization。

