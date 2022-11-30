# Homework 1 - Vector Space Model

## 使用的 tool
Python, nltk, tqdm, re

## 資料前處理

使用stopword, lemmatization，以及正規表達式，將一些特殊符號或數字刪除。

## 模型參數調整

我的 document 和 query 的 term weight 都是使用此公式：

$$tf_{i,j} \times log(1+\frac{N}{n_{i}})$$


## 模型運作原理

主要的概念是先建立出lexicon，並且用建立出的lexicon對應到query和document中，建立向量，也就是把query和document都放到同樣的維度（空間）中後，再計算cosine similarity。

計算Document的TF-IDF值
1) 讀取Document list和Query List的檔案。
2) 文字前處理
4) 計算 Document 的 TF-IDF 值，公式：TF * IDF。
5) 計算 Query 的 TF-IDF 值
計算Cosine similarity
6) 1個Query和1000個Document逐步進行cosine similarity 的計算。再來，利用
sorted方法得到1個Query與1000個Documents的Ranking值。

## 個人心得

一開始一招公式的方法去實作，結果發現每跑一次要花非常久的時間，困擾了很久後，請教別人才知道其實有很多地方是可以省略不用算，也理解到程式能的重要性。尚未優化前，時間花最久的部分在於計算Term-Frequency時，目前也只有1000 document，就要花上40分鐘，幸好後來在網路上看到某位大神，將TF, DF, Lexicon全都放在一個函式內一起算，儲存方式也很不同，受教許多。另外，在看別的人code時，也了解到能好好理解別人的想法，也算是另一種學習。