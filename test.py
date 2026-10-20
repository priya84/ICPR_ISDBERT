import math, torch, operator, time, csv, sys
from transformers import BertTokenizer, BertForNextSentencePrediction, BertForPreTraining, BertConfig
import torch

inputCsvPath = "./OBQA_BERT-NSP/Data/CSQA/train.csv"
outputCsvPath = "./OBQA_BERT-NSP/Data/CSQA/ISD-BERT/train_epoch0.csv"

# inputCsvPath = "./OBQA_BERT-NSP/Data/HellaSwag/train.csv"
# outputCsvPath = "./OBQA_BERT-NSP/Data/HellaSwag/ISD-BERT/train_epoch0.csv"

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')	

########################################################################################################################
# Old Code -- To Combine two BERT Models                                                                               #
########################################################################################################################
# model1 = BertForNextSentencePrediction.from_pretrained('/home/saakur/Desktop/ICPR2022/OBQA_BERT-NSP/bert-base_epoch1')
# inputCsvPath = "/media/4TBHDD/Home_Backup/PSR-VQA/Mac/PSR-VQA/Data/OpenBook/Train_GT/train.csv"
# outputCsvPath = "./OBQA_BERT-NSP/train_epoch2.csv"
# combine = True
# if combine:
# 	model2 = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')	
# 	layers_missed = ['cls.seq_relationship.bias', 'cls.seq_relationship.weight']
# 	for part in layers_missed:
# 		model1.state_dict()[part].data.copy_(model2.state_dict()[part].data)  
# model = model1

########################################################################################################################

model.eval()

with open(inputCsvPath, 'r') as f:
	reader = csv.reader(f)
	# next(reader, None)  # skip the headers
	c = 0
	numQuestions = 0
	numCorrect = 0
	for row in reader:
		numQuestions += 1
		c += 1
		# if c >= 2:
		# 	break
		print(len(row))
		if c == 1:
			with open(outputCsvPath, 'a') as of:
				writer = csv.writer(of)
				writer.writerow(["","video-id","fold-ind","startphrase","sent1","sent2","gold-source","ending0","ending1","ending2","ending3","label"])
				continue
		qid, video_id, fold_ind, startphrase, sent1, sent2, gold_source, ending0, ending1, ending2, ending3, label = row
		text_start = "[CLS] " + sent1 + " [SEP]"
		end0 = ending0 + " [SEP]"
		end1 = ending1 + " [SEP]"
		end2 = ending2 + " [SEP]"
		end3 = ending3 + " [SEP]"

		text0 = text_start + ' ' + end0
		text1 = text_start + ' ' + end1
		text2 = text_start + ' ' + end2
		text3 = text_start + ' ' + end3
	
		tokenized_text0 = tokenizer.tokenize(text0)
		tokenized_text1 = tokenizer.tokenize(text1)
		tokenized_text2 = tokenizer.tokenize(text2)
		tokenized_text3 = tokenizer.tokenize(text3)

		# Convert token to vocabulary indices
		indexed_tokens0 = tokenizer.convert_tokens_to_ids(tokenized_text0)
		indexed_tokens1 = tokenizer.convert_tokens_to_ids(tokenized_text1)
		indexed_tokens2 = tokenizer.convert_tokens_to_ids(tokenized_text2)
		indexed_tokens3 = tokenizer.convert_tokens_to_ids(tokenized_text3)
		
		# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
		segments_ids0 = [0]*(tokenized_text0.index("[SEP]")) + [1]*(len(tokenized_text0) - tokenized_text0.index("[SEP]"))
		segments_ids1 = [0]*(tokenized_text1.index("[SEP]")) + [1]*(len(tokenized_text1) - tokenized_text1.index("[SEP]"))
		segments_ids2 = [0]*(tokenized_text2.index("[SEP]")) + [1]*(len(tokenized_text2) - tokenized_text2.index("[SEP]"))
		segments_ids3 = [0]*(tokenized_text3.index("[SEP]")) + [1]*(len(tokenized_text3) - tokenized_text3.index("[SEP]"))

		# print(segments_ids0, tokenized_text0)
		# print(len(segments_ids0), len(tokenized_text0))

		# Convert inputs to PyTorch tensors
		tokens_tensor0 = torch.tensor([indexed_tokens0])
		tokens_tensor1 = torch.tensor([indexed_tokens1])
		tokens_tensor2 = torch.tensor([indexed_tokens2])
		tokens_tensor3 = torch.tensor([indexed_tokens3])

		segments_tensors0 = torch.tensor([segments_ids0])
		segments_tensors1 = torch.tensor([segments_ids1])
		segments_tensors2 = torch.tensor([segments_ids2])
		segments_tensors3 = torch.tensor([segments_ids3])


		# Predict is Next Sentence ?
		predictions0 = model(tokens_tensor0, token_type_ids=segments_tensors0).logits.cpu().detach().numpy()
		predictions1 = model(tokens_tensor1, token_type_ids=segments_tensors1).logits.cpu().detach().numpy()
		predictions2 = model(tokens_tensor2, token_type_ids=segments_tensors2).logits.cpu().detach().numpy()
		predictions3 = model(tokens_tensor3, token_type_ids=segments_tensors3).logits.cpu().detach().numpy()

		predictions = [predictions0[0][0], predictions1[0][0], predictions2[0][0], predictions3[0][0]]
		
		predictions = sorted(enumerate(predictions), key=operator.itemgetter(1))
		min_index = predictions[3][0]

		if int(min_index) == int(label):
			numCorrect += 1
		if numQuestions%25 == 0:
			print(numCorrect, numQuestions, numCorrect/(numQuestions*1.0))

		with open(outputCsvPath, 'a') as of:
			writer = csv.writer(of)
			writer.writerow([qid, video_id, fold_ind, startphrase, sent1, sent2, gold_source, ending0, ending1, ending2, ending3, int(min_index)])
print(numCorrect, numQuestions, numCorrect/(numQuestions*1.0))
