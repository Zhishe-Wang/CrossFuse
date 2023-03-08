import random
from args import args

from Models import Generator, D_IR, D_VI
import torch
import torch.optim as optim
from torch.autograd import Variable
from loss import g_content_loss
import time
from tqdm import tqdm, trange
import numpy as np
import os
import scipy.io as scio
from utils import make_floor
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def reset_grad(g_optimizer,dir_optimizer,dvi_optimizer):
    dir_optimizer.zero_grad()
    dvi_optimizer.zero_grad()
    g_optimizer.zero_grad()

def train(train_data_ir, train_data_vi):


    models_save_path = make_floor(os.getcwd(), args.save_model_dir)
    print(models_save_path)
    loss_save_path = make_floor(models_save_path,args.save_loss_dir)
    print(loss_save_path)

    G = Generator().cuda()
    D_ir = D_IR().cuda()
    D_vi = D_VI().cuda()
    g_content_criterion = g_content_loss().cuda()


    optimizerG = optim.Adam(G.parameters(), args.g_lr)
    optimizerD_ir = optim.Adam(D_ir.parameters(), args.d_lr)
    optimizerD_vi = optim.Adam(D_vi.parameters(), args.d_lr)

    print("\nG_model : \n", G)
    print("\nD_ir_model : \n", D_ir)
    print("\nD_vi_model : \n", D_vi)

    tbar = trange(args.epochs)

    ir_d_loss_lst = []
    vi_d_loss_lst = []

    g_adversarial_loss_lst=[]
    content_loss_lst = [] 
    all_intensity_loss_lst = []
    all_texture_loss_lst = []
    g_loss_lst = []


    all_ir_d_loss = 0
    all_vi_d_loss = 0

    all_g_adversarial_loss = 0.
    all_content_loss = 0.
    all_intensity_loss = 0.
    all_texture_loss = 0.


    for epoch in tbar:
        print('Epoch %d.....' % epoch)

        G.train()
        D_ir.train()
        D_vi.train()
        batch_size=args.batch_size
        image_set_ir,image_set_vi,batches = utils.load_dataset(train_data_ir,train_data_vi, batch_size,num_imgs=None)

        count = 0


        for batch in range(batches):
            count +=1
            reset_grad(optimizerG, optimizerD_ir,optimizerD_vi)
            img_model = 'L'
            image_paths_ir = image_set_ir[batch * batch_size:(batch * batch_size + batch_size)]
            image_paths_vi = image_set_vi[batch * batch_size:(batch * batch_size + batch_size)]
            img_ir = utils.get_train_images_auto(image_paths_ir, height=args.hight, width=args.width, mode=img_model)
            img_vi = utils.get_train_images_auto(image_paths_vi, height=args.hight, width=args.width, mode=img_model)
            img_ir = Variable(img_ir, requires_grad=False)
            img_vi = Variable(img_vi, requires_grad=False)

            lambda_4 = 10
            img_ir = img_ir.cuda()
            img_vi = img_vi.cuda()


            img_fusion = G(img_ir, img_vi)
            #----------------------------------------------------------
            # (1) Update D_vi network:
            #----------------------------------------------------------

            for _ in range(2):
                D_out_vi= D_vi(img_vi)
                D_loss_vi = - torch.mean(D_out_vi)
                D_out_f = D_vi(img_fusion.detach())
                D_loss_f = D_out_f.mean()


                alpha_vi = torch.rand(img_vi.size(0), 1, 1, 1).cuda().expand_as(img_vi)
                interpolated_vi = Variable(alpha_vi * img_vi.data + (1 - alpha_vi) * img_fusion.data, requires_grad=True)
                Dvi_interpolated = D_vi(interpolated_vi)
                grad_vi = torch.autograd.grad(outputs=Dvi_interpolated,
                                           inputs=interpolated_vi,
                                           grad_outputs=torch.ones(Dvi_interpolated.size()).cuda(),
                                           retain_graph=True,
                                           create_graph=True,
                                           only_inputs=True)[0]
                grad_vi = grad_vi.view(grad_vi.size(0), -1)
                grad_vi_l2norm = torch.sqrt(torch.sum(grad_vi ** 2, dim=1))
                Dvi_penalty= torch.mean((grad_vi_l2norm - 1) ** 2)

                vi_d_loss = D_loss_vi + D_loss_f + Dvi_penalty * lambda_4

                all_vi_d_loss += vi_d_loss.item()

                reset_grad(optimizerG,optimizerD_ir,optimizerD_vi)
                vi_d_loss.backward(retain_graph=True)
                optimizerD_vi.step()

                # ----------------------------------------------------------
                # (2) Update D_ir network: 
                # ----------------------------------------------------------
            for _ in range(2):
                D_out_ir = D_ir(img_ir)
                D_loss_ir = - torch.mean(D_out_ir)

                D_out_f = D_ir(img_fusion.detach())
                D_loss_f = D_out_f.mean()

                alpha_ir = torch.rand(img_ir.size(0), 1, 1, 1).cuda().expand_as(img_ir)
                interpolated_ir = Variable(alpha_ir * img_ir.data + (1 - alpha_ir) * img_fusion.data, requires_grad=True)
                Dir_interpolated = D_ir(interpolated_ir)
                grad_ir = torch.autograd.grad(outputs=Dir_interpolated,
                                              inputs=interpolated_ir,
                                              grad_outputs=torch.ones(Dir_interpolated.size()).cuda(),
                                              retain_graph=True,
                                              create_graph=True,
                                              only_inputs=True)[0]
                grad_ir = grad_ir.view(grad_ir.size(0), -1)
                grad_ir_l2norm = torch.sqrt(torch.sum(grad_ir ** 2, dim=1))
                Dir_penalty = torch.mean((grad_ir_l2norm - 1) ** 2)

                ir_d_loss = D_loss_ir + D_loss_f + Dir_penalty * lambda_4

                all_ir_d_loss += ir_d_loss.item()

                reset_grad(optimizerG, optimizerD_ir,optimizerD_vi)
                ir_d_loss.backward(retain_graph=True)
                optimizerD_ir.step()


            # ----------------------------------------------------------
            # (3) Update G network:
            # ----------------------------------------------------------
            img_fusion = G(img_ir, img_vi)

            content_loss,  intensity_loss , texture_loss = g_content_criterion(img_ir, img_vi,img_fusion)  # models_4
            lambda_1=1
            dir_g_adversarial_loss = -D_ir(img_fusion).mean()
            dvi_g_adversarial_loss = -D_vi(img_fusion).mean()
            g_adversarial_loss = (dir_g_adversarial_loss + dvi_g_adversarial_loss)
            g_loss =lambda_1*g_adversarial_loss +content_loss

            all_intensity_loss += intensity_loss.item()
            all_texture_loss +=texture_loss.item()
            all_g_adversarial_loss += g_adversarial_loss.item()
            all_content_loss += content_loss.item()

            reset_grad(optimizerG, optimizerD_ir, optimizerD_vi)
            g_loss.backward()
            optimizerG.step()

            if (batch + 1) % args.log_interval == 0:
                mesg = "{}\tepoch {}:[{}/{}]\n " \
                       "ir_d_loss: {:.6}\t vi_d_loss: {:.6}" \
                       "\t g_adversarial_loss:{:.6}\t content_loss:{:.6}\t g_loss:{:.6}"  \
                       "\t intensity_loss:{:.6}\t  texture_loss:{:.6}".format(
                    time.ctime(), epoch+1, count, batches,
                    all_ir_d_loss / (args.log_interval), all_vi_d_loss / (args.log_interval),
                    all_g_adversarial_loss / args.log_interval, all_content_loss / args.log_interval,(all_g_adversarial_loss + all_content_loss) / args.log_interval,
                    all_intensity_loss / args.log_interval,  all_texture_loss / args.log_interval
                )
                tbar.set_description(mesg)

                ir_d_loss_lst.append(all_ir_d_loss / args.log_interval)
                vi_d_loss_lst.append(all_vi_d_loss / args.log_interval)

                g_adversarial_loss_lst.append(all_g_adversarial_loss / args.log_interval)
                content_loss_lst.append(all_content_loss / args.log_interval)
                all_intensity_loss_lst.append(all_intensity_loss / args.log_interval)
                all_texture_loss_lst.append(all_texture_loss / args.log_interval)
                g_loss_lst.append((all_g_adversarial_loss + all_content_loss ) / args.log_interval)


                all_ir_d_loss = 0
                all_vi_d_loss = 0

                all_g_adversarial_loss = 0.
                all_content_loss = 0.
                all_intensity_loss = 0
                all_texture_loss = 0



        if (epoch+1) % args.log_iter == 0:
            # SAVE MODELS
            G.eval()
            G.cuda()
            G_save_model_filename = "G_Epoch_" + str(epoch) + ".model"
            G_model_path = os.path.join(models_save_path,G_save_model_filename)
            torch.save(G.state_dict(), G_model_path)


            # SAVE LOSS DATA

            ir_d_loss_part = np.array(ir_d_loss_lst)
            loss_filename_path = "ir_d_loss_epoch_" + str(epoch) + ".mat"
            save_loss_path = os.path.join(loss_save_path, loss_filename_path)
            scio.savemat(save_loss_path, {'ir_d_loss_part': ir_d_loss_part})

            vi_d_loss_part = np.array(vi_d_loss_lst)
            loss_filename_path = "vi_d_loss_epoch_" + str(epoch) + ".mat"
            save_loss_path = os.path.join(loss_save_path, loss_filename_path)
            scio.savemat(save_loss_path, {'vi_d_loss_part': vi_d_loss_part})



            # g_adversarial_loss
            g_adversarial_loss_part = np.array(g_adversarial_loss_lst)
            loss_filename_path = "g_adversarial_loss_epoch_" + str(epoch) + ".mat"
            save_loss_path = os.path.join(loss_save_path, loss_filename_path)
            scio.savemat(save_loss_path, {'g_adversarial_loss_part': g_adversarial_loss_part})

            # content_loss
            content_loss_part = np.array(content_loss_lst)
            loss_filename_path = "content_loss_epoch_" + str(epoch) + ".mat"
            save_loss_path = os.path.join(loss_save_path, loss_filename_path)
            scio.savemat(save_loss_path, {'content_loss_part': content_loss_part})


            all_intensity_loss_part = np.array(all_intensity_loss_lst)
            loss_filename_path = "all_intensity_loss_epoch_" + str(epoch) + ".mat"
            save_loss_path = os.path.join(loss_save_path, loss_filename_path)
            scio.savemat(save_loss_path, {'all_intensity_loss_part': all_intensity_loss_part})



            all_texture_loss_part = np.array(all_texture_loss_lst)
            loss_filename_path = "all_texture_loss_epoch_" + str(epoch) + ".mat"
            save_loss_path = os.path.join(loss_save_path, loss_filename_path)
            scio.savemat(save_loss_path, {'all_texture_loss_part': all_texture_loss_part})


            # g_loss
            g_loss_part = np.array(g_loss_lst)
            loss_filename_path = "g_loss_epoch_" + str(epoch) + ".mat"
            save_loss_path = os.path.join(loss_save_path, loss_filename_path)
            scio.savemat(save_loss_path, {'g_loss_part': g_loss_part})




    # SAVE LOSS DATA
    # d_loss
    ir_d_loss_total = np.array(ir_d_loss_lst)
    loss_filename_path = "ir_d_loss_total_epoch_" + str(epoch) + ".mat"
    save_loss_path = os.path.join(loss_save_path, loss_filename_path)
    scio.savemat(save_loss_path, {'ir_d_loss_total': ir_d_loss_total})

    vi_d_loss_total = np.array(vi_d_loss_lst)
    loss_filename_path = "vi_d_loss_total_epoch_" + str(epoch) + ".mat"
    save_loss_path = os.path.join(loss_save_path, loss_filename_path)
    scio.savemat(save_loss_path, {'vi_d_loss_total': vi_d_loss_total})

    # g_loss
    # g_adversarial_loss
    g_adversarial_loss_total = np.array(g_adversarial_loss_lst)
    loss_filename_path = "g_adversarial_loss_total_epoch_" + str(epoch) + ".mat"
    save_loss_path = os.path.join(loss_save_path, loss_filename_path)
    scio.savemat(save_loss_path, {'g_adversarial_loss_total': g_adversarial_loss_total})

    # content_loss
    content_loss_total = np.array(content_loss_lst)
    loss_filename_path = "content_loss_total_epoch_" + str(epoch) + ".mat"
    save_loss_path = os.path.join(loss_save_path, loss_filename_path)
    scio.savemat(save_loss_path, {'content_loss_total': content_loss_total})


    # all_intensity_loss
    all_intensity_loss_total = np.array(all_intensity_loss_lst)
    loss_filename_path = "all_intensity_loss_total_epoch_" + str(epoch) + ".mat"
    save_loss_path = os.path.join(loss_save_path, loss_filename_path)
    scio.savemat(save_loss_path, {'all_intensity_loss_total': all_intensity_loss_total})

    # all_texture_loss
    all_texture_loss_total = np.array(all_texture_loss_lst)
    loss_filename_path = "all_texture_loss_total_epoch_" + str(epoch) + ".mat"
    save_loss_path = os.path.join(loss_save_path, loss_filename_path)
    scio.savemat(save_loss_path, {'all_texture_loss_total': all_texture_loss_total})


    # g_loss
    g_loss_total = np.array(g_loss_lst)
    loss_filename_path = "g_loss_total_epoch_" + str(epoch) + ".mat"
    save_loss_path = os.path.join(loss_save_path, loss_filename_path)
    scio.savemat(save_loss_path, {'g_loss_total': g_loss_total})


    # SAVE MODELS
    G.eval()
    G.cuda()

    G_save_model_filename = "Final_G_Epoch_" + str(epoch) + ".model"
    G_model_path = os.path.join(models_save_path, G_save_model_filename)
    torch.save(G.state_dict(), G_model_path)

    print("\nDone, trained Final_G_model saved at", G_model_path)





