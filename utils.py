
import numpy as np
import os
from args import args
import scipy.misc

import torch
from PIL import Image
from os import listdir
from os.path import join
from scipy.misc import imread, imsave, imresize
from torchvision import transforms

def load_dataset(ir_imgs_path,vi_imgs_path, BATCH_SIZE, num_imgs=None):
    if num_imgs is None:
        num_imgs = len(ir_imgs_path)
    ir_imgs_path = ir_imgs_path[:num_imgs]
    vi_imgs_path = vi_imgs_path[:num_imgs]
    # random
    mod = num_imgs % BATCH_SIZE
    print('BATCH SIZE %d.' % BATCH_SIZE)
    print('Train images number %d.' % num_imgs)
    print('Train images samples %s.' % str(num_imgs / BATCH_SIZE))

    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        ir_imgs_path =  ir_imgs_path[:-mod]
        vi_imgs_path = vi_imgs_path[:-mod]
    batches = int(len(ir_imgs_path) // BATCH_SIZE)
    return ir_imgs_path,vi_imgs_path, batches

def make_floor(path1,path2):
    path = os.path.join(path1,path2)
    if os.path.exists(path) is False:
        os.makedirs(path)
    return path




def get_train_images_auto(paths, height=args.hight, width=args.width, mode='RGB'):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, mode=mode)
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            image = np.reshape(image, [image.shape[2], image.shape[0], image.shape[1]])
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    images = (images - 127.5) / 127.5

    return images



def prepare_data(directory):
    directory = os.path.join(os.getcwd(), directory)
    images = []
    names = []
    dir = listdir(directory)
    dir.sort()
    for file in dir:
        name = file.lower()
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
        elif name.endswith('.bmp'):
            images.append(join(directory, file))
        elif name.endswith('.tif'):
            images.append(join(directory, file))
        name1 = name.split('.')
        names.append(name1[0])
    return images


def save_feat(index,C,ir_atten_feat,vi_atten_feat,result_path):
    ir_atten_feat = (ir_atten_feat / 2 + 0.5) * 255
    vi_atten_feat = (vi_atten_feat / 2 + 0.5) * 255

    ir_feat_path = make_floor(result_path, "ir_feat")
    index_irfeat_path = make_floor(ir_feat_path, str(index))

    vi_feat_path = make_floor(result_path, "vi_feat")
    index_vifeat_path = make_floor(vi_feat_path, str(index))

    for c in range(C):
        ir_temp = ir_atten_feat[:, c, :, :].squeeze()
        vi_temp = vi_atten_feat[:, c, :, :].squeeze()

        feat_ir = ir_temp.cpu().clamp(0, 255).data.numpy()
        feat_vi = vi_temp.cpu().clamp(0, 255).data.numpy()

        ir_feat_filenames = 'ir_feat_C' + str(c) + '.png'
        ir_atten_path = index_irfeat_path + '/' + ir_feat_filenames
        imsave(ir_atten_path, feat_ir)

        vi_feat_filenames = 'vi_feat_C' + str(c) + '.png'
        vi_atten_path = index_vifeat_path + '/' + vi_feat_filenames
        imsave(vi_atten_path, feat_vi)





def get_image(path, height=args.hight, width=args.width, mode='L'):
    if mode == 'L':
        image = imread(path, mode=mode)
        image = (image - 127.5) / 127.5
        #image = image/255
    elif mode == 'RGB':
        image = Image.open(path).convert('RGB')
    if height is not None and width is not None:
        image = imresize(image, [height, width], interp='nearest')


    return image



def get_test_images(paths, height=None, width=None, mode='RGB'):
    ImageToTensor = transforms.Compose([transforms.ToTensor()])
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, mode=mode)
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            image = ImageToTensor(image).float().numpy()*255
    images.append(image)
    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images

def save_images(path, data):
    if data.shape[2] == 1:
        data = data.reshape([data.shape[0], data.shape[1]])
    imsave(path, data)

def list_images(directory):
    images = []
    names = []
    dir = listdir(directory)
    dir.sort()
    for file in dir:
        name = file.lower()
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
        name1 = name.split('.')
        names.append(name1[0])
    return images