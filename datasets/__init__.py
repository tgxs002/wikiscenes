from torch.utils import data
from .wikiscenes_corr import WikiSegmentation_corr

datasets = {
    'wikiscenes_corr': WikiSegmentation_corr,
}

def get_num_classes(args):
    return datasets[args.dataset.lower()].NUM_CLASS

def get_class_names(args):
    return datasets[args.dataset.lower()].CLASSES

def get_dataloader(args, cfg, split, batch_size=None, test_mode=None):
    dataset_name = args.dataset.lower()
    dataset_cls = datasets[dataset_name]
    if args.export in ['kp_score', 'kp_feature']:
        dataset = dataset_cls(cfg, args.export_set, test_mode)
    else:
        dataset = dataset_cls(cfg, split, test_mode)

    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    shuffle, drop_last = [True, True] if split == 'train' else [False, False]

    if batch_size is None:
        batch_size = cfg.TRAIN.BATCH_SIZE

    return data.DataLoader(dataset, batch_size=batch_size,
                           drop_last=drop_last, shuffle=shuffle, 
                           **kwargs)
