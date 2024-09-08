def set_fusion_hyperparameters(args):
    args.set_defaults(model_name='GCN')
    args.set_defaults(n_epochs=300)
    args.set_defaults(save_result_file='sample.csv')
    args.set_defaults(sweep_name='exp_sample')
    args.set_defaults(ground_metric='euclidean')
    args.set_defaults(activation_mode='mean')
    args.set_defaults(geom_ensemble_type='acts')
    args.set_defaults(sweep_id=21)
    args.set_defaults(ground_metric_normalize='none')
    args.set_defaults(activation_seed=21)
    args.set_defaults(ckpt_type='best')
    args.set_defaults(exact=True)
    args.set_defaults(correction=True)
    args.set_defaults(weight_stats=True)
    args.set_defaults(activation_histograms=True)
    args.set_defaults(prelu_acts=True)
    args.set_defaults(recheck_acc=True)
    args.set_defaults(past_correction=True)
    args.set_defaults(not_squared=True)
    args.set_defaults(handle_skips=True)
    args.set_defaults(fusion_model='sixcnn')
    args.set_defaults(server_gpu=1)
    args.set_defaults(dataset='Cifar10')
    return args

