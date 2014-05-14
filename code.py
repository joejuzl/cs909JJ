from bs4 import BeautifulSoup
import csv
import nltk, numpy, gensim, pickle, cPickle, re, itertools
from nltk.corpus import stopwords
from nltk.stem.porter import *
import nltk.classify
import pprint
from nltk import bigrams
from nltk.tokenize.punkt import PunktWordTokenizer
from gensim import corpora, models, similarities
from collections import defaultdict

pp = pprint.PrettyPrinter(indent=4)

# nltk.download()

path1 = "reuters/reut2-00"
path2 = "reuters/reut2-0"
extension = ".sgm"

# extract SGM text	
def extract(data):
	if isinstance(data, unicode):
		return data.encode('ascii', errors='backslashreplace')
	elif data is None:
		return ""
	else:
		return data.text.encode('ascii', errors='backslashreplace')
	


# print csv		
def toCSV(blop,filename):
	rows = []
	row = []
	row.append("ID")
	for x in range(len(blop[1][0])):
		row.append("C"+str(x))
	row.append("Class")
	i=0
	rows.append(row)
	for item in blop:
		i += 1
		row = []
		row.append(str(i))
		for element in item:
			if(isinstance(element, dict)):
				for a in sorted(element.keys()):
					row.append(str(element[a]))
			if(isinstance(element, str)):
				row.append(element)
		rows.append(row)
	with open(filename+'.csv', 'wb') as f:
		writer = csv.writer(f)
		writer.writerows(rows)
		
# pre process text content
def process(text):
	stemmer = PorterStemmer()
	stop = stopwords.words('english')
	sentences = PunktWordTokenizer().tokenize(text)
	tagged = []
	for sentence in sentences:
		tokens = nltk.word_tokenize(sentence)
		for token in tokens:
			token = stemmer.stem(token)
			token = token.lower()
		tokens = [i for i in tokens if i not in stop and len(i) > 1 and re.match("[a-z]+",i)]		
		#tagged.extend(nltk.pos_tag(tokens))
		tagged.extend(tokens)
	return tagged

# import data and put in dictionary
def loadSGM(fromPickle):
	if fromPickle:
		return cPickle.load(open("docs.p", "rb"))
	collection = {}
	for i in range(0,22):
		print("loading SGM file "+str(i))
		if i < 10:
			file = path1+str(i)+extension
		else:
			file = path2+str(i)+extension
		s = BeautifulSoup(open(file))
		[x.extract() for x in s.findAll(['date','places','people','orgs','exchanges','companies','unknown'])]
		all = s.findAll("reuters")
		size = len(all)
		for j in range(size):
			temp = all[j]
			doc = {}
			doc['cgisplit']= extract(temp['cgisplit'])
			doc['lewissplit']= extract(temp['lewissplit'])
			doc['newid']= extract(temp['newid'])
			doc['oldid']= extract(temp['oldid'])
			doc['topics'] = []
			topics = temp.topics.findAll("d")
			for x in range(len(topics)):
				doc['topics'].append(extract(topics[x]))
			doc['title'] = extract(temp.find('title'))
			doc['author'] = extract(temp.find('author'))
			doc['body'] = extract(temp.find('body'))
			doc['text'] = extract(temp.find('text'))
			collection[int(doc['newid'])-1] = doc
	cPickle.dump(collection, open("docs.p", "wb" ))
	return collection

#process data
def processData(data, num, fromPickle):
	if fromPickle:
		dataset = cPickle.load(open("data.p", "rb"))
		return dict(itertools.islice(dataset.iteritems(), 0, num)) 
	pdocs = {}
	for i in range(num):
		data[i]['body'] = process(data[i]['body'])
		data[i]['title'] = process(data[i]['title'])
		data[i]['text'] = process(data[i]['text'])
		pdocs[i] = data[i]
		print("processing doc "+str(i))		
	cPickle.dump(pdocs, open("data.p", "wb" ))
	return pdocs

# build corpus
def buildCorpus(data):
	corpus = []
	for i in range(len(data)):
		topics = data[i]['topics']
		text = data[i]['text']	
		corpus.append(text)	
	return corpus

#extract LDA topics as features	
def extractLDAFeatureSet(data, numberOfFeatures,  topTen, fromPickle):
	if fromPickle:
		return cPickle.load(open("ldaTrain.p", "rb")),cPickle.load(open("ldaTest.p", "rb"))
	corpus = buildCorpus(data)	
	dictionary = corpora.Dictionary(corpus)
	raw_corpus = [dictionary.doc2bow(t) for t in corpus]		
	trainset = []
	testset = []
	train_corpus = []
	test_corpus = []
	dataTrain = []
	dataTest = []
	for i in range(len(raw_corpus)):
		type = data[i]['lewissplit']
		if type == "NOT-USED":
			continue
		elif type == "TRAIN":
			train_corpus.append(raw_corpus[i])
			dataTrain.append(data[i])
		else:
			test_corpus.append(raw_corpus[i])
			dataTest.append(data[i])
	ldaTrain = gensim.models.ldamodel.LdaModel(corpus=train_corpus, id2word=dictionary, num_topics=numberOfFeatures)
	featureTuplesTrain = list(ldaTrain[train_corpus])
	featureTuplesTest = list(ldaTrain[test_corpus])
	for i in range(len(featureTuplesTrain)):
		topics = dataTrain[i]['topics']
		if len(topics) < 1:
			continue
		flag = False
		for t in topics:
			if t in topTen:
				topic = t
				flag = True
				break;
		if flag == False:
			continue
		d = {}
		features = featureTuplesTrain[i]
		for k, v in features:
			d[k] = v
		for k in range(numberOfFeatures):
			if not k in d:
				d[k] = 0
		trainset.append((d,topic))
	
	for i in range(len(featureTuplesTest)):
		topics = dataTest[i]['topics']
		if len(topics) < 1:
			continue
		flag = False
		for t in topics:
			if t in topTen:
				topic = t
				flag = True
				break;
		if flag == False:
			continue
		d = {}
		features = featureTuplesTest[i]
		for k, v in features:
			d[k] = v
		for k in range(numberOfFeatures):
			if not k in d:
				d[k] = 0
		testset.append((d,topic))
	
	cPickle.dump(trainset, open("ldaTrain.p", "wb" ))
	cPickle.dump(testset, open("ldaTest.p", "wb" ))
	return trainset,testset
	
def extractUnigramFeatureSet(data, numberOfFeatures, topTen, fromPickle):
	if fromPickle:
		return cPickle.load(open("uniTrain.p", "rb")),cPickle.load(open("uniTest.p", "rb"))
	corpus = buildCorpus(data)	
	# create dictionary
	dictionary = corpora.Dictionary(corpus)
	dictionary.filter_extremes(keep_n=numberOfFeatures)
	raw_corpus = [dictionary.doc2bow(t) for t in corpus]
	trainset = []
	testset = []
	for i in range(len(data)):
		topics = data[i]['topics']
		type = data[i]['lewissplit']
		if type == "NOT-USED":
			continue
		if len(topics) < 1:
			continue
		flag = False
		for t in topics:
			if t in topTen:
				topic = t
				flag = True
				break;
		if flag == False:
			continue	
		l = raw_corpus[i]
		d = {}
		for k, v in l:
			d[k] = v
		for k in range(len(dictionary)):
			if not k in d:
				d[k] = 0
		if type == "TRAIN":
			trainset.append((d,topic))
		else:
			testset.append((d,topic))		
	cPickle.dump(trainset, open("uniTrain.p", "wb" ))
	cPickle.dump(testset, open("uniTest.p", "wb" ))
	return trainset,testset
	
def extractBigramFeatureSet(data, numberOfFeatures, topTen, fromPickle):
	if fromPickle:
		return cPickle.load(open("biTrain.p", "rb")),cPickle.load(open("biTest.p", "rb"))
	corpus = buildCorpus(data)
	for x in range(len(corpus)):
		bi = bigrams(corpus[x])
		for i in range(len(bi)):
			t1,t2 = bi[i]
			bi[i] = t1+t2
		corpus[x] = bi	
	# create dictionary
	dictionary = corpora.Dictionary(corpus)
	dictionary.filter_extremes(keep_n=numberOfFeatures)
	raw_corpus = [dictionary.doc2bow(t) for t in corpus]
	trainset = []
	testset = []
	for i in range(len(data)):
		topics = data[i]['topics']
		type = data[i]['lewissplit']
		if type == "NOT-USED":
			continue
		if len(topics) < 1:
			continue
		flag = False
		for t in topics:
			if t in topTen:
				topic = t
				flag = True
				break;
		if flag == False:
			continue	
		l = raw_corpus[i]
		d = {}
		for k, v in l:
			d[k] = v
		for k in range(len(dictionary)):
			if not k in d:
				d[k] = 0	
		if type == "TRAIN":
			trainset.append((d,topic))
		else:
			testset.append((d,topic))		
	cPickle.dump(trainset, open("biTrain.p", "wb" ))
	cPickle.dump(testset, open("biTest.p", "wb" ))
	return trainset,testset

def extractTfidfFeatureSet(data, numberOfFeatures, topTen, fromPickle):
	if fromPickle:
		return cPickle.load(open("uniTrain.p", "rb")),cPickle.load(open("uniTest.p", "rb"))
	corpus = buildCorpus(data)	
	# create dictionary
	dictionary = corpora.Dictionary(corpus)
	dictionary.filter_extremes(keep_n=numberOfFeatures)
	raw_corpus = [dictionary.doc2bow(t) for t in corpus]
	tfidf = models.TfidfModel(raw_corpus)
	trainset = []
	testset = []
	for i in range(len(data)):
		topics = data[i]['topics']
		type = data[i]['lewissplit']
		if type == "NOT-USED":
			continue
		if len(topics) < 1:
			continue
		flag = False
		for t in topics:
			if t in topTen:
				topic = t
				flag = True
				break;
		if flag == False:
			continue	
		l = tfidf[raw_corpus[i]]
		d = {}
		for k, v in l:
			d[k] = v
		for k in range(len(dictionary)):
			if not k in d:
				d[k] = 0
		if type == "TRAIN":
			trainset.append((d,topic))
		else:
			testset.append((d,topic))		
	cPickle.dump(trainset, open("uniTrain.p", "wb" ))
	cPickle.dump(testset, open("uniTest.p", "wb" ))
	return trainset,testset
	
def popClasses(data,num):
	topics = {}
	for i in range(len(data)):
		topicList = data[i]['topics']
		for topic in topicList:
			if topic in topics:
				topics[topic] = topics[topic]+1
			else:
				topics[topic] = 0
	top = sorted(topics, key=topics.get, reverse=True)[:num]
	return top
	

	
	
docs = processData(loadSGM(True), 21578, True)


topTopics = popClasses(docs,10)


#extract features

#trainset, testset = extractUnigramFeatureSet(docs, 50, topTopics, False)
#trainset, testset = extractTfidfFeatureSet(docs, 50, topTopics, False)
#trainset, testset = extractBigramFeatureSet(docs, 50, topTopics, False)
trainset, testset = extractLDAFeatureSet(docs,10, topTopics, False)


toCSV(trainset,"ldaTrain")
toCSV(testset,"ldaTest")



