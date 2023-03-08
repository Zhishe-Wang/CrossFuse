import time
from train import train
from generate import generate
from args import args
import utils
import torch
import os
from Models import Generator

flag =0
if flag == 1:
    IS_TRAINING = True
else:
    IS_TRAINING = False

def load_model(model_path):
    G_model = Generator()
    G_model.load_state_dict(torch.load(model_path))
    print('# generator parameters:', sum(param.numel() for param in G_model.parameters()))
    G_model.eval()
    G_model.cuda()
    return G_model


def main():
    if IS_TRAINING:
        data_dir_ir = utils.list_images(args.train_ir)
        data_dir_vi = utils.list_images(args.train_vi)
        train_data_ir = data_dir_ir
        train_data_vi = data_dir_vi

        print("\ntrain_data_ir num is ", len(train_data_ir))
        print("\ntrain_data_vi num is ", len(train_data_vi))

        train(train_data_ir, train_data_vi)

        # testing
    else:
        print("\nBegin to generate pictures ...\n")

        model_name = 'Final_G_Epoch_13.model'

        test_imgs_path= "./test_imgs/tno/"
        print('TNO date set begin to test')
        result = "results"
        model_path = os.path.join(os.getcwd(), 'models_training', model_name)
        with torch.no_grad():
            model = load_model(model_path)
            model.eval()
            model.cuda()
            begin = time.time()
            for i in range(25):
                    index = i + 1
                    ir_path = test_imgs_path+ "IR" + str(index) + ".png"
                    vis_path = test_imgs_path + "VIS" + str(index) + ".png"
                    generate(model, ir_path, vis_path, model_path, index, mode='L')
            end = time.time()
            print("consumption time of generating:%s " % (end - begin))





if __name__ == "__main__":
    main()