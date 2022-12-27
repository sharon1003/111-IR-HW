# HW6_61147006S.py

import numpy as np
import gensim
from gensim.summarization import bm25
from gensim.models import word2vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm  # 計算時間的工具條
import re
import time
import nltk
from nltk.corpus import words
nltk.download('omw-1.4')                # use for lammatize
nltk.download('wordnet')
nltk.download('words')
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

def tokenizer(text): # 切開文字
    return text.split()

def preprocessor(text):  # 移除特殊字元，僅保留英數字
    filters = re.compile(u'[^0-9a-zA-Z]+', re.UNICODE)
    return filters.sub(' ', text) #remove special characters

def lemmatize(word):
    lemma = lemmatizer.lemmatize(word,'v')
    if lemma == word:
        lemma = lemmatizer.lemmatize(word,'n')
    return lemma

def lemma(text):
    return [lemmatize(token) for token in text.split()]


# Function returning vector reperesentation of a document
def get_embedding_w2v(doc_tokens):
	embeddings = []
	if len(doc_tokens) < 1:
		return np.zeros(vector_size)
	else:
		for tok in doc_tokens:
			if tok in model.wv.vocab.keys():
				embeddings.append(model.wv.get_vector(tok))
				# print('1')
			else:
				embeddings.append(np.random.rand(vector_size))
				# print('2')
		# mean the vectors of individual words to get the vector of the document
		return np.mean(embeddings, axis=0)


document_path = 'data/documents/'
document_list = open('data/docs_id_list.txt', 'r')
query_path = 'data/queries/'
query_list = open('data/queries_id_list.txt', 'r')
documents = document_list.read().split('\n')
query = query_list.read().split('\n')

doc_texts = []
corpus = []
with open("stopwords", "r", encoding='utf-8') as fin:
    stopwords = fin.read()
    for doc_id in tqdm(documents):
        file = open(document_path + doc_id + '.txt', 'r', encoding="utf8")
        article_text = file.read()
        # preprocessing
        temp = []
        for word in lemma(preprocessor(article_text.strip())):
            if word.lower() in stopwords or len(word) > 13:
                continue
            temp.append(word.lower())
        doc = ' '.join(temp)
        corpus.append(temp)
        doc_texts.append(doc)

query_texts = []
bm25_que = []
with open("stopwords", "r", encoding='utf-8') as fin:
    stopwords = fin.read()
    for q_id in tqdm(query):
        file = open(query_path + q_id + '.txt', 'r', encoding="utf8")
        article_text = file.read()
        # preprocessing
        temp = []
        for word in lemma(preprocessor(article_text.strip())):
            if word.lower() in stopwords or len(word) > 13:
                continue
            temp.append(word.lower())
        que = ' '.join(temp)
        bm25_que.append(temp)
        query_texts.append(que)

print("BM25")
bm25Model = bm25.BM25(corpus) # 利用corpus做訓練

print('Skipgram')
seed = 666
sg = 1 # sg=1 -> skip-gram, sg=0 -> cbow
window_size = 4		# 向前或向後看幾個字
vector_size = 500     # 向量維度
min_count = 1		# Ignores all words with total frequency lower than this.
workers = 8
epochs = 50
batch_words = 10000

print(f"w2v model start at {time.strftime('%X')}")
model = word2vec.Word2Vec(
	corpus,
	min_count=min_count,
	size=vector_size,
	workers=workers,
	iter=epochs,
	window=window_size,
	sg=sg,
	seed=seed,
	batch_words=batch_words
)
model.save('word2vec_hw6.model')
print(f"w2v model end at {time.strftime('%X')}")


# Load model
print(f"Load Model start at {time.strftime('%X')}")
print("Loading...")
model = word2vec.Word2Vec.load("word2vec_hw6.model")
print(f"Load Model end at {time.strftime('%X')}")

# # Zero padding
padding = np.zeros((model.vector_size,), dtype=np.float32)


w2v_score = {}
t = 1
# query preprocessing
with open('skipgram_hw6.csv', 'w', encoding='utf-8') as fout:
	fout.write('Query,RetrievedDocuments\n')
	for q_id, que_tokens in zip(query, bm25_que):
		que_vec = get_embedding_w2v(que_tokens)
		print('Times', t)	
		print(f"Time {time.strftime('%X')}")
		t += 1	
		bm25_score = bm25Model.get_scores(que_tokens) # (1, doc_len)
		tmp_score = {}
		k = 0
		# document preprocessing
		for doc_id, doc_tokens in zip(documents, corpus):
			doc_vec = get_embedding_w2v(doc_tokens)	

			skipgram = cosine_similarity(que_vec.reshape(1, -1), doc_vec.reshape(1, -1))[0][0]		
			tmp_score[doc_id] = skipgram + bm25_score[k]
			k += 1
		

		sorted_value = sorted(tmp_score.items(), key=lambda x: x[1], reverse = True)
		w2v_score[q_id] = sorted_value[:10000]

		for key, values in w2v_score.items():
			ans = key + ","
			for v in values:
				ans = ans + v[0] + ' '
			ans = ans.strip() + '\n'
		fout.write(ans)