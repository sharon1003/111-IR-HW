# IR_HW3

# 輸入input組數
times = int(input())

score = []
for i in range(times):
	# each query have __ documents, type=list
	return_documents = input().split()
	relevant_ans = input().split()

	doc_len = len(return_documents)
	rele_len = len(relevant_ans)
	rele_index = []

	for rele_doc in relevant_ans:
		if rele_doc not in return_documents:
			rele_index.append(doc_len+1)
		else:
			rele_index.append(return_documents.index(rele_doc))

	rele_index.sort() # sort from 小到大 reverse=False

	tmp_score = []
	for i, index in enumerate(rele_index):
		if index == (doc_len+1): # 代表此rele沒有被system找到
			tmp_score.append(0)
		else:
			tmp_score.append( (i + 1) / (index + 1) )
	score.append(sum(tmp_score) / rele_len)

print(round(sum(score) / times, 4))

