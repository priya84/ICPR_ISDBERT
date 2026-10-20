import numpy as np
import sys, os, csv

inputNPY = sys.argv[1]
inputCSV = sys.argv[2]
outputCSV = sys.argv[3]

inFile = np.load(inputNPY, allow_pickle=True)
predLabels = np.argmax(inFile[0], axis=-1)

# inputCsvPath = "/media/4TBHDD/Home_Backup/PSR-VQA/Mac/PSR-VQA/Data/OpenBook/Train_GT/train.csv"
# outputCsvPath = "./OBQA_BERT-NSP/train_epoch2.csv"

with open(inputCSV, 'r') as f:
	reader = csv.reader(f)
	c = -2
	for row in reader:
		c += 1
		if c == -1:
			with open(outputCSV, 'a') as of:
				writer = csv.writer(of)
				writer.writerow(["","video-id","fold-ind","startphrase","sent1","sent2","gold-source","ending0","ending1","ending2","ending3","label"])
				continue
		qid, video_id, fold_ind, startphrase, sent1, sent2, gold_source, ending0, ending1, ending2, ending3, label = row
		with open(outputCSV, 'a') as of:
			writer = csv.writer(of)
			writer.writerow([qid, video_id, fold_ind, startphrase, sent1, sent2, gold_source, ending0, ending1, ending2, ending3, int(predLabels[c])])