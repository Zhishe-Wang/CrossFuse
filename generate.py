from args import args
import utils
from torch.autograd import Variable
from Models import Generator
import torch
import os
from utils import make_floor
from scipy.misc import imsave

def _generate_fusion_image(G_model, ir_img, vis_img):

    f = G_model(ir_img, vis_img)
    return f

def load_model(model_path):
    G_model = Generator()
    G_model.load_state_dict(torch.load(model_path))
    print('# generator parameters:', sum(param.numel() for param in G_model.parameters()))
    G_model.eval()
    G_model.cuda()
    return G_model

def generate(model,ir_path, vis_path, result,  index,  mode):
    result = "results"
    ir_img = utils.get_test_images(ir_path, mode=mode)
    vis_img = utils.get_test_images(vis_path, mode=mode)
    ir_img = ir_img.cuda()
    vis_img = vis_img.cuda()
    ir_img = Variable(ir_img, requires_grad=False)
    vis_img = Variable(vis_img, requires_grad=False)



    img_fusion = _generate_fusion_image(model, ir_img, vis_img)
    img_fusion = (img_fusion / 2 + 0.5) * 255
    img_fusion = img_fusion.squeeze()
    img_fusion = img_fusion

    if args.cuda:
        img = img_fusion.cpu().clamp(0, 255).data.numpy()
    else:
        img = img_fusion.clamp(0, 255).data[0].numpy()

    result_path = make_floor(os.getcwd(),result)

    if index<10:
        f_filenames = "100" + str(index) + '.png'
        output_path = result_path + '/'+ f_filenames
        imsave(output_path, img)

    else:
        f_filenames = "10" + str(index) + '.png'
        output_path = result_path + '/'+ f_filenames
        imsave(output_path, img)

