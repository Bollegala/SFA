"""
Create a bipartite graph as described in the following paper.

@inproceedings{Pan:WWW:2010,
	Author = {Sinno Jialin Pan and Xiaochuan Ni and Jian-Tao Sun and Qiang Yang and Zheng Chen},
	Booktitle = {WWW 2010},
	Title = {Cross-Domain Sentiment Classification via Spectral Feature Alignment},
	Year = {2010}}

Danushka Bollegala. 
2013/09/25
"""

import sys
import math
import numpy as np
import scipy.io as sio 
import features

def generateFeatureVectors(domain):
	"""
	Create feature vectors for each review in the domain. 
	"""
	FeatGen = features.FEATURE_GENERATOR()
	for (mode, label) in [("train", "positive"), ("train", "negative"), ("train", "unlabeled"),
							("test", "positive"), ("test", "negative")]:
		fname = "../../reviews/%s-data/%s/%s.tagged" % (mode, domain, label)
		fvects = FeatGen.process_file(fname, label)
		writeFeatureVectorsToFile(fvects, "../work/%s/%s.%s" % (domain, mode, label))	
	pass


def writeFeatureVectorsToFile(fvects, fname):
	"""
	Write each feature vector in fvects in a single line in fname. 
	"""
	F = open(fname, 'w')
	for e in fvects:
		for w in e[1].keys():
			F.write("%s " % w)
		F.write("\n")
	F.close()
	pass


def getCounts(S, M, fname):
	"""
	Get the feature co-occurrences in the file fname and append 
	those to the dictionary M. We only consider features in S.
	"""
	count = 0
	F = open(fname)
	for line in F:
		count += 1
		#if count > 1000:
		#	break
		allP = line.strip().split()
		p = []
		for w in allP:
			if w in S:
				p.append(w) 
		n = len(p)
		for i in range(0,n):
			for j in range(i + 1, n):
				pair = (p[i], p[j])
				rpair = (p[j], p[i])
				if pair in M:
					M[pair] += 1
				elif rpair in M:
					M[rpair] += 1
				else:
					M[pair] = 1
	F.close()
	pass


def getVocab(S, fname):
	"""
	Get the frequency of each feature in the file named fname. 
	"""
	F = open(fname)
	for line in F:
		p = line.strip().split()
		for w in p:
			S[w] = S.get(w, 0) + 1
	F.close()
	pass


def selectTh(h, t):
	"""
	Select all elements of the dictionary h with frequency greater than t. 
	"""
	p = {}
	for (key, val) in h.iteritems():
		if val > t:
			p[key] = val
	del(h)
	return p


def getVal(x, y, M):
	"""
	Returns the value of the element (x,y) in M.
	"""
	if (x,y) in M:
		return M[(x,y)] 
	elif (y,x) in M:
		return M[(y,x)]
	else:
		return 0
	pass


def createMatrix(source, target):
	"""
	Read the unlabeled data (test and train) for both source and the target domains. 
	Compute the full co-occurrence matrix. Drop co-occurrence pairs with a specified
	minimum threshold. For a feature w, compute its score(w),
	score(w) = {\sum_{x \in S} pmi(w, x)} + {\sum_{y \in T} pmi(w, y)}
	and sort the features in the descending order of their scores. 
	Write the co-occurrence matrix to a file with name source-target.cooc (fid, fid, cooc) and the 
	scores to a file with name source-target.pmi (feat, fid, score).
	"""
	# Parameters
	if source == "dvd":
		SourceFreqTh = 50
	else:
		SourceFreqTh = 50

	if target == "dvd":
		TargetFreqTh = 50
	else:
		TargetFreqTh = 50

	coocTh = 1
	noPivots = 500
	noSourceSpecific = 500
	noTargetSpecific = 500

	print "Source = %s, Target = %s" % (source, target)

	# Get the set of source domain features.
	S = {}
	getVocab(S, "../work/%s/train.positive" % source)
	getVocab(S, "../work/%s/train.negative" % source)
	getVocab(S, "../work/%s/train.unlabeled" % source)
	print "Total source features =", len(S)
	# Remove source domain features with total frequency less than SourceFreqTh
	S = selectTh(S, SourceFreqTh)
	print "After thresholding at %d we have = %d" % (SourceFreqTh, len(S))

	# Get the set of target domain features.
	T = {}
	#getVocab(T, "../work/%s/train.positive" % target) # labels not used.
	#getVocab(T, "../work/%s/train.negative" % target) # labels not used.
	getVocab(T, "../work/%s/train.unlabeled" % target)
	print "Total target features =", len(T)
	# Remove target domain features with total frequency less than TargetFreqTh
	T = selectTh(T, TargetFreqTh)
	print "After thresholding at %d we have = %d" % (TargetFreqTh, len(T))	

	# Compute the co-occurrences of features in reviews
	src_M = {}
	tgt_M = {}
	print "Source Vocabulary size =", len(S)
	getCounts(S, src_M, "../work/%s/train.positive" % source)
	print "%s positive %d" % (source, len(src_M)) 
	getCounts(S, src_M, "../work/%s/train.negative" % source)
	print "%s negative %d" % (source, len(src_M))
	getCounts(S, src_M, "../work/%s/train.unlabeled" % source)
	print "%s unlabeled %d" % (source, len(src_M))

	#getCounts(T, tgt_M, "../work/%s/train.positive" % target)
	#print "%s positive %d" % (target, len(M))	
	#getCounts(T, tgt_M, "../work/%s/train.negative" % target)
	#print "%s negative %d" % (target, len(M))	
	getCounts(T, tgt_M, "../work/%s/train.unlabeled" % target)
	print "%s unlabeled %d" % (target, len(tgt_M))	

	# Remove co-occurrence less than the coocTh
	src_M = selectTh(src_M, coocTh)
	tgt_M = selectTh(tgt_M, coocTh)

	# Compute the intersection of source and target domain features.
	pivots = set(S.keys()).intersection(set(T.keys()))
	print "Total no. of pivots =", len(pivots)

	labels = {'books':'book', "electronics":"electronic", "dvd":"dvd", "kitchen":"kitchen"}

	# Compute PMI scores for pivots.
	C = {}
	src_N = sum(S.values())
	tgt_N = sum(T.values())
	i = 0
	for pivot in pivots:
		C[pivot] = 0.0
		i += 1
		val = getVal(pivot, labels[source], src_M)
		C[pivot] += 0 if (val < coocTh) else getPMI(val, S[labels[source]], S[pivot], src_N)	
		val = getVal(pivot, labels[target], tgt_M)	
		C[pivot] += 0 if (val < coocTh) else getPMI(val, T[labels[target]], T[pivot], tgt_N)
		if i % 500 == 0:
			print "%d: pivot = %s, MI = %.4g" % (i, pivot, C[pivot])
	pivotList = C.items()
	pivotList.sort(lambda x, y: -1 if x[1] > y[1] else 1)
	# write pivots to a file.
	pivotsFile = open("../work/%s-%s/DI_list" % (source, target), 'w')
	DI = []
	for (i, (w, v)) in enumerate(pivotList[:noPivots]):
		pivotsFile.write("%d %s P %s\n" % (i+1, w, str(v))) 
		DI.append(w)
	pivotsFile.close()

	# Select source specific features
	srcSpecificFeatures = {}
	featCount = 0
	for w in S:
		featCount += 1
		if w not in T:
			srcSpecificFeatures[w] = 0.0
			for x in S:
				if x is not w:
					val = getVal(w, x, src_M)
					srcSpecificFeatures[w] += 0 if (val < coocTh) else getPMI(val, S[w], S[x], src_N)
		print "SS %d of %d %s %f" % (featCount, len(S), w, srcSpecificFeatures.get(w, 0))

	# Select target specific features
	tgtSpecificFeatures = {}
	featCount = 0
	for w in T:
		featCount += 1
		if w not in S:
			tgtSpecificFeatures[w] = 0.0
			for x in T:
				if x is not w:
					val = getVal(w, x, tgt_M)
					tgtSpecificFeatures[w] += 0 if (val < coocTh) else getPMI(val, T[w], T[x], tgt_N)
		print "TS %d of %d %s %f" % (featCount, len(T), w, tgtSpecificFeatures.get(w, 0))

	srcList = srcSpecificFeatures.items()
	srcList.sort(lambda x, y: -1 if x[1] > y[1] else 1)
	tgtList = tgtSpecificFeatures.items()
	tgtList.sort(lambda x, y: -1 if x[1] > y[1] else 1)

	# Domain specific feature list.
	DSFile = open("../work/%s-%s/DS_list.%d" % (source, target, (noSourceSpecific + noTargetSpecific)), 'w')
	DS = []
	count = 0
	for (f,v) in srcList[:noSourceSpecific]:
		count += 1
		DS.append(f)
		DSFile.write("%d %s S %s\n" % (count, f, str(v)))
	for (f,v) in tgtList[:noTargetSpecific]:
		count += 1
		DS.append(f)
		DSFile.write("%d %s T %s\n" % (count, f, str(v)))
	DSFile.close() 

	# Compute matrix DSxDS and save it.
	nDS = len(DS)
	RS = np.zeros((nDS, nDS), dtype=np.float)
	for i in range(0, nDS):
		for j in range(i+1, nDS):
			val = getVal(DS[i], DS[j], src_M) + getVal(DS[i], DS[j], tgt_M)
			if val > coocTh:
				RS[i,j] = RS[j,i] = val
	print "Writing DSxDS.mat...",
	sio.savemat("../work/%s-%s/DSxDS.mat" % (source, target), {'DSxDS':RS})
	print "Done"

	# Compute matrix DIxDI and save it.
	nDI = len(DI)
	RI = np.zeros((nDI, nDI), dtype=np.float)
	for i in range(0, nDI):
		for j in range(i+1, nDI):
			val = getVal(DI[i], DI[j], src_M) + getVal(DI[i], DI[j], tgt_M)
			if val > coocTh:
				RI[i,j] = RI[j,i] = val
	print "Writing DIxDI.mat...",
	sio.savemat("../work/%s-%s/DIxDI.mat" % (source, target), {'DIxDI':RI})
	print "Done"

	# Compute matrix DSxSI and save it. 
	R = np.zeros((nDS, nDI), dtype=np.float)
	for i in range(0, nDS):
		for j in range(0, nDI):
			val = getVal(DS[i], DI[j], src_M) +  getVal(DS[i], DI[j], tgt_M)
			if val > coocTh:
				R[i,j] = val
	print "Writing DSxDI.mat...",
	sio.savemat("../work/%s-%s/DSxDI.mat" % (source, target), {'DSxDI':RI})
	print "Done"
	pass


def getPMI(n, x, y, N):
	"""
	Compute the weighted PMI value. 
	"""
	pmi =  math.log((float(n) * float(N)) / (float(x) * float(y)))
	res = pmi * (float(n) / float(N))
	return 0 if res < 0 else res


def generateAll():
	"""
	Generate matrices for all pairs of domains. 
	"""
	domains = ["books", "electronics", "dvd", "kitchen"]
	for source in domains:
		for target in domains:
			if source == target:
				continue
			createMatrix(source, target)
	pass


if __name__ == "__main__":
	#generateFeatureVectors("books")
	#generateFeatureVectors("dvd")
	#generateFeatureVectors("electronics")
	#generateFeatureVectors("kitchen")
	generateAll()
	#createMatrix("dvd", "books")
	