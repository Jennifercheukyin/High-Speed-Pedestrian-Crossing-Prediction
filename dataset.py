import os
import numpy as np
import torch 
import torchvision
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from action_predict import *
from jaad_data import *
from PIL import Image, ImageDraw 
from torch.utils.data import DataLoader
from utils import *


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
        self.poses = get_pose(self.data_raw['data']['image'], 
                                self.data_raw['data']['ped_id'], 
                                data_type=self.data_type,
                                file_path='data/features/jaad/poses',
                                dataset='jaad')
        # use bounding box crop data
        
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
        labels = self.data_raw['data']['crossing'][idx]
        bbox = self.data_raw['data']['box_org'][idx]  # 'bbox': list([x1, y1, x2, y2])
        poses = self.poses[idx]
        poses = np.reshape(poses, (poses.shape[0], 18, 2))
        speed = self.data_raw['data']['speed'][idx] # (16,1)
        # cordinates = bbox[0]
        # bw, bh = cordinates[2] - cordinates[0], cordinates[3] - cordinates[1]
        # print(bbox)
        # fig, ax = plt.subplots()

        # imp = img_paths[0]
        # im = Image.open(imp)
        # im_crop = im.crop(cordinates)
        # img = self.transform(im_crop)
        # img = img.cpu().detach().numpy()
        # img = img.transpose(1,2,0)
        # ax.imshow(img)
        # # rect = patches.Rectangle((cordinates[0], cordinates[1]), bw, bh, linewidth=1, edgecolor='r', facecolor='none')
        # # ax.add_patch(rect)

        # # plot pose 
        # pose = poses[0]
        # plt.scatter(pose[:, 0] * 224, pose[:, 1] * 224)

        # plt.show()

        
        # read img from paths and transform img path to image of size (3,224,224)
        img_seq = []
        for imp, coordinates in zip(img_paths, bbox): # img = './JAAD/images/video_0001/00491.png 
            img = Image.open(imp)
            img = img.crop(coordinates)
            img = self.transform(img)
            img = torch.squeeze(img, axis=0)
            img = img.detach().numpy()
            img_seq.append(img)

        img_seq = torch.Tensor(img_seq) # tensor (16,3,224,224)
        labels = torch.Tensor(labels)
        poses = torch.Tensor(poses)
        speed = torch.Tensor(speed)

        sigmoid = torch.nn.Sigmoid()
        speed = sigmoid(speed)

        return img_seq, labels, poses, speed

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
        #   pass
        #   # imdb = PIE(data_path=os.environ.copy()['PIE_PATH'])
        # elif self.configs['model_opts']['dataset'] == 'jaad':
        #   # if use docker:
        #   # imdb = JAAD(data_path=os.environ.copy()['JAAD_PATH'])

        #   # if use local path
        #   self.imdb = JAAD(data_path='./JAAD/')

    
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
        assert model_opts['obs_length'] ==  16
        model_opts['normalize_boxes'] = False
        # self._generator = model_opts.get('generator', False)
        # data_type_sizes_dict = {}
        # process = model_opts.get('process', True)
        data, neg_count, pos_count = self.method_class.get_data_sequence(data_type, data_raw, model_opts)

        return {'data': data,
                'ped_id': data['ped_id'],
                'image': data['image'],
                'tte': data['tte'],
                'count': {'neg_count': neg_count, 'pos_count': pos_count}}


if __name__ == "__main__": 
    train_dataset = JAADDataset('train', 'MASK_PCPA_jaad_2d')
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)


    img_seq, labels, poses, speed = train_dataset.__getitem__(30)
    print(img_seq.shape)
    print(labels.shape)
    print(poses.shape)
    print(speed)

    # train_dataset.__getitem__(0)
    # poses = get_pose(train_dataset.data_raw['data']['image'], 
    #               train_dataset.data_raw['data']['ped_id'], 
    #               data_type='train',
    #                 file_path='data/features/jaad/poses',
    #                 dataset='jaad')
    # print(poses.shape)
