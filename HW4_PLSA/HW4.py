# version4.py

import numpy as np
import re
from tqdm import tqdm  # 計算時間的工具條
import nltk
from nltk.corpus import words
import time
import numba
from numba.typed import List,Dict
from numba import types
np.random.seed(1331) 					# set the random seed let each time have the same result
nltk.download('omw-1.4')				# use for lammatize
nltk.download('wordnet')
nltk.download('words')
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

def read_doc_file():
    with open("data/docs_id_list.txt", 'r') as fin:
        doc_list = fin.read().split('\n')  # list 型態
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
        doc_list = fin.read().split('\n')  # list 型態
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

def word_count(doc_dict, is_doc=False):
	with open("stopwords", "r", encoding='utf-8') as fin:
		print('TF')
		stopwords = fin.read() # str

		list_TF = []
		Corpus_dict = {}
		Corpus_index = 0
		BackGround = {}

		for doc_id, sentence in tqdm(doc_dict.items()):
			tf = {}
			sent_list = lemma(sentence)
			# tmp_lexicon = set(list_lexicon_o).intersection(set(sent_list))
			for word in sent_list:
				if word not in stopwords:
					if word not in Corpus_dict:
						Corpus_dict[word] = Corpus_index
						Corpus_index += 1

					if word not in tf:
						tf[word] = 1
					else:
						tf[word] += 1

					if word not in BackGround:
						BackGround[word] = 1
					else:
					 	BackGround[word] += 1

			list_TF.append(tf)
		
		if is_doc:
			return Corpus_dict, list_TF, BackGround
		else:
			return Corpus_dict, list_TF

def dict_TF_toList(dict_wordIndex, list_TF):
	length = len(dict_wordIndex)
	list_newTF = np.zeros((len(list_TF),length))
	for index, dict_doc in enumerate(list_TF):
		tempTF = np.zeros(length)
		for Term in dict_doc.keys():
			tempTF[dict_wordIndex[Term]] = dict_doc[Term]
		list_newTF[index] = tempTF

	return list_newTF

def BG_to_list(Corpus, BackGround):
    tempBG = {}
    for Term in BackGround:
        tempBG[Corpus[Term]] = BackGround[Term]

    for Term in tqdm(tempBG):
        tempBG[Term] /= sum(tempBG.values())

    return tempBG

def init(i_topicNum, i_docNum, i_wordNum):
	p_WT = np.random.rand(i_wordNum, i_topicNum)
	for topic in range(i_topicNum):
		p_WT[:, topic] /= p_WT[:, topic].sum()

	p_TD = np.random.rand(i_topicNum, i_docNum)
	for doc in range(i_docNum):
		p_TD[:, doc] /= p_TD[:, doc].sum()

	return p_WT, p_TD

@numba.jit()
def E_step(p_WT, p_TD, i_topicNum, index_topic, index_word, index_doc):
	denominator = 0
	for topic in range(i_topicNum):
		denominator += (p_WT[index_word][topic] * p_TD[topic][index_doc])

	if denominator == 0:
		print(index_topic, index_word, index_doc)

	p_twd = (p_WT[index_word][index_topic] * p_TD[index_topic][index_doc]) / denominator

	return p_twd

@numba.jit()
def M_step(p_WT, p_TD, i_topicNum, i_wordNum, i_docNum, list_DocToWord):
	# update p(W|T)
	for index_topic in range(i_topicNum):
		for index_word in range(i_wordNum):
			update = 0
			for index_doc in range(i_docNum):
				if list_DocToWord[index_doc][index_word] != 0:
					update += list_DocToWord[index_doc][index_word] * E_step(p_WT, p_TD, i_topicNum, index_topic, index_word, index_doc)
			p_WT[index_word][index_topic] = update
		p_WT[:, index_topic] /= summation(p_WT[:, index_topic]) + 1e-20

	# update p(T|D)
	for index_doc in range(i_docNum):
		for index_topic in range(i_topicNum):
			update = 0
			for index_word in range(i_wordNum):
				if list_DocToWord[index_doc][index_word] != 0:
					update += list_DocToWord[index_doc][index_word] * E_step(p_WT, p_TD, i_topicNum, index_topic, index_word, index_doc)
			update /= summation(list_DocToWord[index_doc]) + 1e-20
			p_TD[index_topic][index_doc] = update


	return p_WT, p_TD

@numba.jit()
def summation(list):
	all_sum = 0
	for i in range(len(list)):
		all_sum += list[i] 
	return all_sum

@numba.jit()
def likelihood(p_WT, p_TD, i_docNum, i_wordNum, i_topicNum, list_DocToWord):
	loss = 0
	for index_doc in range(i_docNum):
		for index_word in range(i_wordNum):
			if list_DocToWord[index_doc][index_word] != 0:
				temp_loss = 0
				for index_topic in range(i_topicNum):
					temp_loss += p_WT[index_word][index_topic] * p_TD[index_topic][index_doc]
					
				loss += np.log(temp_loss) * list_DocToWord[index_doc][index_word]
	return loss

def EM_Algorithm(epochs, old_WT, old_TD, i_topicNum, i_docNum, i_wordNum, list_DocToWord):
	print(f"start at {time.strftime('%X')}")
	for epoch in range(epochs):
		p_WT, p_TD = M_step(old_WT, old_TD, i_topicNum, i_wordNum, i_docNum, list_DocToWord)
		old_WT, old_TD = p_WT, p_TD
		print("Epoch: ", epoch)
		print(f"M_step end at {time.strftime('%X')}")
		loss = likelihood(p_WT, p_TD, i_docNum, i_wordNum, i_topicNum, list_DocToWord)
		print("loss: ", loss)

		np.save('p_WT_7_2', p_WT)
		np.save('p_TD_7_2', p_TD)

	return p_WT, p_TD

# new query
def query_doc(query_dict, doc_list, Corpus_dict, p_WT, p_TD, i_topicNum, i_docNum, i_wordNum, list_DocToWord):
	with open('ans_PLSA.csv', 'w', encoding='utf-8') as fout:
		with open("stopwords", "r", encoding='utf-8') as fin:
			stopwords = fin.read() # str
			fout.write('Query,RetrievedDocuments\n')

			alpha = 0.33
			beta = 0.33

			score = {}
			for query_id, query in tqdm(query_dict.items()):
				tmp_score = {}
				for index_doc in range(i_docNum):
					result = 1
					for word in lemma(query):
						if word in Corpus_dict.keys():
							index_word = Corpus_dict[word] # get word index
							if list_DocToWord[index_doc][index_word] != 0:
								p_wd = list_DocToWord[index_doc][index_word] / summation(list_DocToWord[index_doc])
								p_twd = np.sum(p_WT[index_word, :] * p_TD[:, index_doc])
								p_bg = list_BG[index_word]
								result = result * ((alpha * p_wd) + (beta * p_twd) + (1 - alpha - beta) * p_bg)
							else:
								p_twd = np.sum(p_WT[index_word, :] * p_TD[:, index_doc])
								p_bg = list_BG[index_word]
								result = result * ((beta * p_twd) + (1 - alpha - beta) * p_bg)

					tmp_score[doc_list[index_doc]] = result

				sorted_value = sorted(tmp_score.items(), key=lambda x: x[1], reverse = True)
				# print(sorted_value)
				score[query_id] = sorted_value[:4000]

				for key, values in score.items():
					ans = key + ","
					for v in values:
						ans = ans + v[0] + ' '
				ans = ans.strip() + '\n'
				fout.write(ans)


doc_dict, doc_list = read_doc_file()
query_dict, query_list = read_query_file()
Corpus_dict, list_TF, BackGround = word_count(doc_dict, True)
list_DocToWord = dict_TF_toList(Corpus_dict, list_TF) # return word_document frequency
list_BG = BG_to_list(Corpus_dict, BackGround)  # get each word's P(Wi | BG)
print(np.shape(list_DocToWord))
# initialization
K = 8							# the amount of topics
N = len(list_DocToWord)			# the amount of documents
M = len(list_DocToWord[0])		# the amount of words
epochs = 50						# iteration times
p_WT, p_TD = init(K, N, M)
new_WT, newTD = EM_Algorithm(epochs, p_WT, p_TD, K, N, M, list_DocToWord)
query_doc(query_dict, doc_list, Corpus_dict, new_WT, newTD, K, N, M, list_DocToWord)