import os
import numpy as np
import torch 
import torchvision
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from action_predict import *
from jaad_data import *
from PIL import Image 
from torch.utils.data import DataLoader


class JAADDataset(torch.utils.data.Dataset): 
	def __init__(self, data_type, model_name):
		self.data_type = data_type # whether it is train or test 
		self.data_raw = None # all data info, including images, ped_ids, bbox, crossing... 
		self.mode_name = model_name
		self.config_file = 'config_files/ours/' + model_name + '.yaml' # config file path
		self.configs_default ='config_files/configs_default.yaml' # default config file path 
		self.configs = None
		self.model_configs = None
		self.imdb = JAAD(data_path='./JAAD/')
		self.method_class = None

		# get data sequence 
		self.readConfigFile() 
		beh_seq_train = self.imdb.generate_data_trajectory_sequence(self.data_type, **self.configs['data_opts'])
		self.method_class = action_prediction(self.configs['model_opts']['model'])(**self.configs['net_opts'])
		self.data_raw = self.get_data(self.data_type, beh_seq_train, {**self.configs['model_opts'], 'batch_size': 2})
		
		self.transform = transforms.Compose([transforms.Resize((224,224)), 
										transforms.ToTensor(), 
										transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


	def __getitem__(self, idx):
		"""
		Args: 
			idx: id of video sequence to get 
		Return: 
			A dictionary containing a video sequence of index idx 
		"""
		img_paths = self.data_raw['data']['image'][idx] # (16,1)
		ped_ids = self.data_raw['data']['ped_id'][idx]
		label = self.data_raw['data']['crossing'][idx]
		
		img_seq = []
		img_features_seq = []

		for imp, p in zip(img_paths, ped_ids):
			# img = './JAAD/images/video_0001/00491.png'

			# transform img path to image of size (3,224,224)
			img = Image.open(imp)
			img = self.transform(img)
			img = torch.squeeze(img, axis=0)
			img = img.detach().numpy()
			img_seq.append(img)

			# get pkl file to facilitate training
			if(self.data_type == 'train'): 
				set_id = imp.split('/')[-3]  
				vid_id = imp.split('/')[-2]
				img_name = imp.split('/')[-1].split('.')[0]
				img_save_folder = os.path.join('data/features/jaad/local_context_cnn_vgg_raw_1.5', set_id, vid_id)
				img_save_path = os.path.join(img_save_folder, img_name + '_' + p[0] + '.pkl')
				with open(img_save_path, 'rb') as fid:
				    try:
				        img_features = pickle.load(fid)
				    except:
				        img_features = pickle.load(fid, encoding='bytes')
				img_features_seq.append(img_features)
			
		img_seq = torch.Tensor(img_seq) # tensor (16,3,224,224)
		label = torch.Tensor(label)
		img_features_seq = torch.Tensor(img_features_seq) # TODO: needs to be normalized 
	
		return img_seq, label, img_features_seq

	def __len__(self): 
		return self.data_raw['data']['crossing'].shape[0] # 2134

	def readConfigFile(self): 
		print(self.config_file)
		# Read default Config file
		with open(self.configs_default, 'r') as f:
			self.configs = yaml.safe_load(f)

		with open(self.config_file, 'r') as f: 
			self.model_configs = yaml.safe_load(f)

		# Update configs based on the model configs
		for k in ['model_opts', 'net_opts']:
			if k in self.model_configs:
				self.configs[k].update(self.model_configs[k])

		# Calculate min track size
		tte = self.configs['model_opts']['time_to_event'] if isinstance(self.configs['model_opts']['time_to_event'], int) else \
			self.configs['model_opts']['time_to_event'][1]
		self.configs['data_opts']['min_track_size'] = self.configs['model_opts']['obs_length'] + tte

		# update model and training options from the config file
		dataset = self.model_configs['exp_opts']['datasets']
		self.configs['data_opts']['sample_type'] = 'beh' if 'beh' in dataset else 'all'
		self.configs['model_opts']['overlap'] = 0.6 if 'pie' in dataset else 0.8
		self.configs['model_opts']['dataset'] = dataset.split('_')[0]
		self.configs['train_opts']['batch_size'] = self.model_configs['exp_opts']['batch_size']
		self.configs['train_opts']['lr'] = self.model_configs['exp_opts']['lr']
		self.configs['train_opts']['epochs'] = self.model_configs['exp_opts']['epochs']

		model_name = self.configs['model_opts']['model']
		# Remove speed in case the dataset is jaad
		if 'RNN' in model_name and 'jaad' in dataset:
			self.configs['model_opts']['obs_input_type'] = self.configs['model_opts']['obs_input_type']

		for k, v in self.configs.items():
			print(k,v)

		# set batch size
		if model_name in ['ConvLSTM']:
			self.configs['train_opts']['batch_size'] = 2
		if model_name in ['C3D', 'I3D']:
			self.configs['train_opts']['batch_size'] = 4
		if model_name in ['PCPA']:
			self.configs['train_opts']['batch_size'] = 1
		if 'MultiRNN' in model_name:
			self.configs['train_opts']['batch_size'] = 8
		if model_name in ['TwoStream']:
			self.configs['train_opts']['batch_size'] = 16

		# if self.configs['model_opts']['dataset'] == 'pie':
		# 	pass
		# 	# imdb = PIE(data_path=os.environ.copy()['PIE_PATH'])
		# elif self.configs['model_opts']['dataset'] == 'jaad':
		# 	# if use docker:
		# 	# imdb = JAAD(data_path=os.environ.copy()['JAAD_PATH'])

		# 	# if use local path
		# 	self.imdb = JAAD(data_path='./JAAD/')

	
	def get_data(self, data_type, data_raw, model_opts):
		"""
		Generates data train/test/val data
		Args:
			data_type: Split type of data, whether it is train, test or val
			data_raw: Raw tracks from the dataset
			model_opts: Model options for generating data
		Returns:
			A dictionary containing, data, data parameters used for model generation,
			effective dimension of data (the number of rgb images to be used calculated accorfing
			to the length of optical flow window) and negative and positive sample counts
		"""

		print('Enter MASK_PCPA_4_2D get_data')
		assert model_opts['obs_length']	==	16
		model_opts['normalize_boxes'] = False
		# self._generator = model_opts.get('generator', False)
		# data_type_sizes_dict = {}
		# process = model_opts.get('process', True)
		dataset = model_opts['dataset']
		data, neg_count, pos_count = self.method_class.get_data_sequence(data_type, data_raw, model_opts)

		return {'data': data,
				'ped_id': data['ped_id'],
				'image': data['image'],
				'tte': data['tte'],
				'count': {'neg_count': neg_count, 'pos_count': pos_count}}


if __name__ == "__main__": 
	train_dataset = JAADDataset('train', 'MASK_PCPA_jaad_2d')
	train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
	
	img_seq, label, ped_ids = train_dataset.__getitem__(0)
	print(img_seq.shape)
	print(ped_ids.shape)
