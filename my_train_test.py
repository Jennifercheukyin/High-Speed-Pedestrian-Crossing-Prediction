from dataset import JAADDataset
from PIL import Image 
from torchvision import transforms, utils
from action_predict import *
from jaad_data import *
import torch
import cv2
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from my_model import MyModel 
import pdb


def train_test(epoch):

	train_dataset = JAADDataset('train', 'MASK_PCPA_jaad_2d')
	train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True)
	test_dataset = JAADDataset('test', 'MASK_PCPA_jaad_2d')
	test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, drop_last=True)

	model = MyModel().cuda()
	weight = torch.Tensor([1760.0/2134.0, 1-1760.0/2134.0]).cuda() 
	criterion = nn.CrossEntropyLoss(weight=weight)
	optimizer = optim.Adam(model.parameters(), lr=0.01)

	for e in range(epoch): 

		train_running_loss = 0.0
		train_running_acc = 0.0
		test_running_loss = 0.0
		test_running_acc = 0.0

		model.train()
		for i, data in enumerate(train_dataloader): 
			train_img_seq, train_labels, train_extra_features = data
			train_img_seq = train_img_seq.cuda()
			train_labels = train_labels.cuda().long().squeeze()

			optimizer.zero_grad()
			h0 = torch.zeros(2,4,512).cuda() # (n_layers * n_directions, batch_size, hidden_size)
			train_outputs = model(train_img_seq, h0)
			# prediction = outputs.detach() > 0.5
			prediction = torch.softmax(train_outputs.detach(), dim=1)[:,1] > 0.5
			prediction = prediction * 1.0 
			# pdb.set_trace()
			correct = (prediction == train_labels.float()) * 1.0

			loss = criterion(train_outputs, train_labels)
			acc = correct.sum() / train_labels.shape[0]
			train_running_loss += loss.item()
			train_running_acc += acc.item()

			loss.backward()
			optimizer.step()
			
			if (i + 1) % 10 == 0:    
				print('Train loss: ', train_running_loss / (10 * ((i + 1) / 10)))
				print('Train acc: ', train_running_acc / (10 * ((i + 1) / 10)))

		model.eval()
		for i, data in enumerate(test_dataloader): 
			test_img_seq, test_labels, test_extra_features = data
			test_img_seq = test_img_seq.cuda()
			test_labels = test_labels.cuda().long().squeeze()

			h0 = torch.zeros(2,4,512).cuda()
			test_outputs = model(test_img_seq, h0)
			prediction = torch.softmax(test_outputs.detach(), dim=1)[:,1] > 0.5
			prediction = prediction * 1.0 
			# pdb.set_trace()
			correct = (prediction == test_labels.float()) * 1.0

			loss = criterion(test_outputs, test_labels)
			acc = correct.sum() / test_labels.shape[0]
			test_running_loss += loss.item()
			test_running_acc += acc.item()

			if (i + 1) % 10 == 0:    
				print('Test loss: ', test_running_loss / (10 * ((i + 1) / 10)))
				print('Test acc: ', test_running_acc / (10 * ((i + 1) / 10)))

		avg_train_loss = train_running_loss / len(train_dataloader)
		avg_train_acc = train_running_acc / len(train_dataloader)
		avg_test_loss = test_running_loss / len(test_dataloader)
		avg_test_acc = test_running_acc / len(test_dataloader)
		print("Train loss: ",  avg_train_loss)
		print("Train accuracy: ",  avg_train_acc)
		print("Test loss: ",  avg_test_loss)
		print("Test accuracy: ", avg_test_acc)


if __name__ == "__main__": 
	train_test(epoch=2)
