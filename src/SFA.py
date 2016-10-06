"""
Peform Spectral Feature Alignment for Cross-Domain Sentiment Classification.

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
import scipy.sparse as sp
from sparsesvd import sparsesvd
import subprocess

import features

def trainLBFGS(train_file, model_file):
    """
    Train lbfgs on train file. and evaluate on test file.
    Read the output file and return the classification accuracy.
    """
    retcode = subprocess.call(
        "classias-train -tb -a lbfgs.logistic -pc1=0 -pc2=1 -m %s %s > /dev/null"  %\
        (model_file, train_file), shell=True)
    return retcode


def testLBFGS(test_file, model_file):
    """
    Evaluate on the test file.
    Read the output file and return the classification accuracy.
    """
    output = "../work/output"
    retcode = subprocess.call("cat %s | classias-tag -m %s -t > %s" %\
                              (test_file, model_file, output), shell=True)
    F = open(output)
    accuracy = 0
    correct = 0
    total = 0
    for line in F:
        if line.startswith("Accuracy"):
            p = line.strip().split()
            accuracy = float(p[1])
    F.close()
    return accuracy


def generateFeatureVectors(domain):
    """
    Create feature vectors for each review in the domain. 
    """
    FeatGen = features.FEATURE_GENERATOR()
    for (mode, label) in [("train", "positive"), ("train", "negative"), ("train", "unlabeled"),
                            ("test", "positive"), ("test", "negative")]:
        fname = "../reviews/%s-data/%s/%s.tagged" % (mode, domain, label)
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
        #   break
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
    domainTh = {'books':20, 'dvd':100, 'kitchen':20, 'electronics':20}
    SourceFreqTh = domainTh[source]
    TargetFreqTh = domainTh[target]
    coocTh = 5
    noPivots = 500

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
    getVocab(T, "../work/%s/train.positive" % target) # labels not used.
    getVocab(T, "../work/%s/train.negative" % target) # labels not used.
    getVocab(T, "../work/%s/train.unlabeled" % target)
    print "Total target features =", len(T)
    # Remove target domain features with total frequency less than TargetFreqTh
    T = selectTh(T, TargetFreqTh)
    print "After thresholding at %d we have = %d" % (TargetFreqTh, len(T))

    # Get the union (and total frequency in both domains) for all features.
    V = S.copy()
    for w in T:
        V[w] = S.get(w, 0) + T[w]

    # Compute the co-occurrences of features in reviews
    M = {}
    print "Vocabulary size =", len(V)
    getCounts(V, M, "../work/%s/train.positive" % source)
    print "%s positive %d" % (source, len(M)) 
    getCounts(V, M, "../work/%s/train.negative" % source)
    print "%s negative %d" % (source, len(M))
    getCounts(V, M, "../work/%s/train.unlabeled" % source)
    print "%s unlabeled %d" % (source, len(M))
    getCounts(V, M, "../work/%s/train.positive" % target)
    print "%s positive %d" % (target, len(M))   
    getCounts(V, M, "../work/%s/train.negative" % target)
    print "%s negative %d" % (target, len(M))   
    getCounts(V, M, "../work/%s/train.unlabeled" % target)
    print "%s unlabeled %d" % (target, len(M))  
    # Remove co-occurrence less than the coocTh
    M = selectTh(M, coocTh)

    # Compute the intersection of source and target domain features.
    pivots = set(S.keys()).intersection(set(T.keys()))
    print "Total no. of pivots =", len(pivots)

    # Compute PMI scores for pivots.
    C = {}
    N = sum(V.values())
    i = 0
    for pivot in pivots:
        C[pivot] = 0.0
        i += 1
        for w in S:
            val = getVal(pivot, w, M)
            C[pivot] += 0 if (val < coocTh) else getPMI(val, V[w], V[pivot], N)
        for w in T: 
            val = getVal(pivot, w, M)
            C[pivot] += 0 if (val < coocTh) else getPMI(val, V[w], V[pivot], N)
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

    DSwords = set(S.keys()).union(set(T.keys())) - pivots 
    DSList = list(DSwords)
    print "Total no. of domain specific features =", len(DSList)

    # Domain specific feature list.
    DSFile = open("../work/%s-%s/DS_list" % (source, target), 'w')
    count = 0
    for w in DSList:
        count += 1
        DSFile.write("%d %s\n" % (count, w))
    DSFile.close() 
    nDS = len(DSList)
    nDI = len(DI)

    # Compute matrix DSxSI and save it. 
    R = np.zeros((nDS, nDI), dtype=np.float)
    for i in range(0, nDS):
        for j in range(0, nDI):
            val = getVal(DSList[i], DI[j], M)
            if val > coocTh:
                R[i,j] = val
    print "Writing DSxDI.mat...",
    sio.savemat("../work/%s-%s/DSxDI.mat" % (source, target), {'DSxDI':R})
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


def learnProjection(sourceDomain, targetDomain):
    """
    Learn the projection matrix and store it to a file. 
    """
    h = 50 # no. of latent dimensions.
    print "Loading the bipartite matrix...",
    coocData = sio.loadmat("../work/%s-%s/DSxDI.mat" % (sourceDomain, targetDomain))
    M = sp.lil_matrix(coocData['DSxDI'])
    (nDS, nDI) = M.shape
    print "Done."
    print "Computing the Laplacian...",
    D1 = sp.lil_matrix((nDS, nDS), dtype=np.float64)
    D2 = sp.lil_matrix((nDI, nDI), dtype=np.float64)
    for i in range(0, nDS):
        D1[i,i] = 1.0 / np.sqrt(np.sum(M[i,:].data[0]))
    for i in range(0, nDI):
        D2[i,i] = 1.0 / np.sqrt(np.sum(M[:,i].T.data[0]))
    B = (D1.tocsr().dot(M.tocsr())).dot(D2.tocsr())
    print "Done."
    print "Computing SVD...",
    ut, s, vt = sparsesvd(B.tocsc(), h)
    sio.savemat("../work/%s-%s/proj.mat" % (sourceDomain, targetDomain), {'proj':ut.T})
    print "Done."    
    pass


def evaluate_SA(source, target, project):
    """
    Report the cross-domain sentiment classification accuracy. 
    """
    gamma = 1.0
    print "Source Domain", source
    print "Target Domain", target
    if project:
        print "Projection ON", "Gamma = %f" % gamma
    else:
        print "Projection OFF"
    # Load the projection matrix.
    M = sp.csr_matrix(sio.loadmat("../work/%s-%s/proj.mat" % (source, target))['proj'])
    (nDS, h) = M.shape
    # Load the domain specific features.
    DSfeat = {}
    DSFile = open("../work/%s-%s/DS_list" % (source, target))
    for line in DSFile:
        p = line.strip().split()
        DSfeat[p[1].strip()] = int(p[0])
    DSFile.close()
    # write train feature vectors.
    trainFileName = "../work/%s-%s/trainVects.SFA" % (source, target)
    testFileName = "../work/%s-%s/testVects.SFA" % (source, target)
    featFile = open(trainFileName, 'w')
    count = 0
    for (label, fname) in [(1, 'train.positive'), (-1, 'train.negative')]:
        F = open("../work/%s/%s" % (source, fname))
        for line in F:
            count += 1
            #print "Train ", count
            words = set(line.strip().split())
            # write the original features.
            featFile.write("%d " % label)
            x = sp.lil_matrix((1, nDS), dtype=np.float64)
            for w in words:
                #featFile.write("%s:1 " % w)
                if w in DSfeat:
                    x[0, DSfeat[w] - 1] = 1
            # write projected features.
            if project:
                y = x.tocsr().dot(M)
                for i in range(0, h):
                    featFile.write("proj_%d:%f " % (i, gamma * y[0,i])) 
            featFile.write("\n")
        F.close()
    featFile.close()
    # write test feature vectors.
    featFile = open(testFileName, 'w')
    count = 0
    for (label, fname) in [(1, 'test.positive'), (-1, 'test.negative')]:
        F = open("../work/%s/%s" % (target, fname))
        for line in F:
            count += 1
            #print "Test ", count
            words = set(line.strip().split())
            # write the original features.
            featFile.write("%d " % label)
            x = sp.lil_matrix((1, nDS), dtype=np.float64)
            for w in words:
                #featFile.write("%s:1 " % w)
                if w in DSfeat:
                    x[0, DSfeat[w] - 1] = 1
            # write projected features.
            if project:
                y = x.dot(M)
                for i in range(0, h):
                    featFile.write("proj_%d:%f " % (i, gamma * y[0,i])) 
            featFile.write("\n")
        F.close()
    featFile.close()
    # Train using classias.
    modelFileName = "../work/%s-%s/model.SFA" % (source, target)
    trainLBFGS(trainFileName, modelFileName)
    # Test using classias.
    acc = testLBFGS(testFileName, modelFileName)
    print "Accuracy =", acc
    print "###########################################\n\n"
    return acc


def batchEval():
    """
    Evaluate on all 12 domain pairs. 
    """
    resFile = open("../work/batchSFA.csv", "w")
    domains = ["books", "electronics", "dvd", "kitchen"]
    resFile.write("Source, Target, NoProj, Proj\n")
    for source in domains:
        for target in domains:
            if source == target:
                continue
            createMatrix(source, target)
            learnProjection(source, target)
            resFile.write("%s, %s, %f, %f\n" % (source, target, 
                evaluate_SA(source, target, False), evaluate_SA(source, target, True)))
            resFile.flush()
    resFile.close()
    pass

if __name__ == "__main__":
    source = "books"
    target = "dvd"
    #generateFeatureVectors("books")
    #generateFeatureVectors("dvd")
    #generateFeatureVectors("electronics")
    #generateFeatureVectors("kitchen")
    #generateAll()
    #createMatrix(source, target)
    #learnProjection(source, target)
    #evaluate_SA(source, target, False)
    #evaluate_SA(source, target, True)
    batchEval()
    