class args():
    epochs = 14
    batch_size = 4

    datasets_prepared = False

    train_ir = 'E:\clip picture last\ir'
    train_vi = 'E:\clip picture last/vi'


    hight = 256
    width = 256
    image_size = 256

    save_model_dir = "models_training"
    save_loss_dir = "loss"

    cuda = 1

    g_lr = 0.0001
    d_lr = 0.0004
    log_interval = 5
    log_iter = 1