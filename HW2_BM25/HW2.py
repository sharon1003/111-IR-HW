import numpy as np
import re
from tqdm import tqdm  # 計算時間的工具條
import nltk
from nltk.corpus import words
nltk.download('omw-1.4')				# use for lammatize
nltk.download('wordnet')
nltk.download('words')
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()


def read_doc_file():
	with open("data/docs_id_list.txt", 'r') as fin:
		doc_list = fin.read().split('\n') 		# list 型態
		
		all_docs = {}
		for doc in doc_list:	
			path = "data/documents/" + doc + ".txt"
			f = open(path, 'r', encoding='utf-8')
			tmp = f.read()
			all_docs[doc] = preprocessor(tmp.strip())
			f.close()

		return all_docs, doc_list

def read_query_file():
	with open("data/queries_id_list.txt", 'r') as fin:
		doc_list = fin.read().split('\n') 		# list 型態
		
		all_docs = {}
		for doc in doc_list:	
			path = "data/queries/" + doc + ".txt"
			f = open(path, 'r', encoding='utf-8')
			tmp = f.read()
			all_docs[doc] = preprocessor(tmp.strip())
			f.close()

		return all_docs, doc_list

def tokenizer(text): # 切開文字
	return text.split()

def preprocessor(text):  # 移除特殊字元，僅保留英數字
	filters = re.compile(u'[^a-zA-Z]+', re.UNICODE)
	return filters.sub(' ', text) #remove special characters

def lemmatize(word):
	lemma = lemmatizer.lemmatize(word,'v')
	if lemma == word:
		lemma = lemmatizer.lemmatize(word,'n')
	return lemma

def lemma(text):
	return [lemmatize(token) for token in text.split()]

def tf_df_function(doc_dict, is_doc=False): # input: doc_dict and bag_of_word, output: np.array()
	with open("stopwords", "r", encoding='utf-8') as fin:
		stopwords = fin.read() # str

		TF_dict = {}
		df_dict = {}
		doc_len = {}
		lexicon = []

		tmp2 = 0
		for doc_id, sentence in tqdm(doc_dict.items()):
			tf = {}
			sent_list = lemma(sentence)
			tmp1 = 0
			for word in sent_list:
				if word.lower() in stopwords or len(word) > 13:
					continue
				lexicon.append(word) # lexicon
				tmp2 += 1
				if word not in tf:
					tf[word] = 1
					tmp1 += 1
				else:
					tf[word] += 1
					tmp1 += 1

				# tokenzier cost a lot of time, it is better to count at once
				if is_doc:
					if word in df_dict:
						if doc_id not in df_dict[word]:
							df_dict[word].append(doc_id)
					else:
						df_dict[word] = [doc_id]
			doc_len[doc_id] = tmp1
			TF_dict[doc_id] = tf

		avg = tmp2 / len(doc_dict)
		lexicon = list(set(lexicon))

		if is_doc:
			return TF_dict, df_dict, doc_len, avg, lexicon
		else:
			return TF_dict, lexicon


def idf_function(N, df_dict): # input: 1. total number of doc 2. df_dict
	print('Start idf')
	idf = {}
	for word, df_times in tqdm(df_dict.items()):
		idf[word] = np.log((N - len(df_times) + 0.5) / (len(df_times) + 0.5))

	return idf

def tfidf_function(tf_dict, idf_dict):
	print('Start tfidf')
	tfidf = {}
	for doc_id, word_dict in tqdm(tf_dict.items()):
		tmp_tfidf = {}
		for word, tf in word_dict.items():
			tmp_tfidf[word] = tf * idf_dict[word] if word in idf_dict else 0
		tfidf[doc_id] = tmp_tfidf

	return tfidf

def parameterS3():
	k3 = 1.8
	s3 = k3 + 1
	return s3, k3

def parameterS1_bm25(doc_len, doc_avg):
	k1 = 1.8
	b = 0.75
	k1 = k1 * ( (1-b) + b * (doc_len / doc_avg) )
	s1 = k1 + 1
	return s1, k1

def BM25(q_tf, doc_tf, query_vocab, idf_dict, doc_len, doc_avg):
	s1, k1 = parameterS1_bm25(doc_len, doc_avg)
	s3, k3 = parameterS3()

	tmp = []
	for word in query_vocab:
		try:
			doc_tf[word]
		except:
			doc_tf[word] = 0

		term1 = (s3 * doc_tf[word]) / (k1 + doc_tf[word])
		term2 = (s1 * q_tf[word]) / (k1 + q_tf[word])
		term3 = idf_dict[word]
		tmp.append(term1 * term2 * term3)

	return sum(tmp)



doc_dict, doc_list = read_doc_file()   		# return doc_dict and doc_list
query_dict, query_list = read_query_file()  # return query_dict and query_list

N = len(doc_dict)
doc_tf, df_dict, doc_len, doc_avg, lexicon1 = tf_df_function(doc_dict, True) # return doc_tf, df, doc's length
doc_idf = idf_function(N, df_dict)						# return idf
que_tf, lexicon2 = tf_df_function(query_dict)					# return query_tf

doc_tf_idf = tfidf_function(doc_tf, doc_idf)
que_tf_idf = tfidf_function(que_tf, doc_idf)

print('Lexicon', len(lexicon1))
doc_avg = int(doc_avg)


with open('ans.csv', 'w', encoding='utf-8') as fout:
	with open("stopwords", "r", encoding='utf-8') as fin:
		stopwords = fin.read() # str
		print("Start BM25")
		fout.write('Query,RetrievedDocuments\n')

		bm25_score = {}
		for q_id, query_sentence in tqdm(query_dict.items()):

			query_vocab = []
			# 拆開出query的數量
			for word in lemma(query_sentence):
				if word not in lexicon1:
					continue
				if word in stopwords:
					continue
				if word not in query_vocab:
					query_vocab.append(word)

			tmp_cos = {}
			for doc_id, tf in doc_tf_idf.items():
				tmp_cos[doc_id] = BM25(que_tf[q_id], doc_tf[doc_id], query_vocab, doc_idf, doc_len[doc_id], doc_avg)

			sorted_value = sorted(tmp_cos.items(), key=lambda x: x[1], reverse = True)

			bm25_score[q_id] = sorted_value[:1500]

			for key, values in bm25_score.items():
				ans = key + ","
				for v in values:
					ans = ans + v[0] + ' '
				ans = ans.strip() + '\n'
			fout.write(ans)




