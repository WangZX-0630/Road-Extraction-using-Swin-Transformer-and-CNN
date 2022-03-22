from datasets import data_SpaceNet, data_LoveDA

def build_train_dataset(args):
    if args.dataset_name == 'SpaceNet':
        train_dataset = data_SpaceNet.Segmentation(args, base_dir=args.dataset_basedir, split='train')
        return train_dataset
    elif args.dataset_name == 'LoveDA':
        train_dataset = data_LoveDA.Segmentation(args, base_dir=args.dataset_basedir, split='train')
        return train_dataset

def build_val_dataset(args):
    if args.dataset_name == 'SpaceNet':
        val_dataset = data_SpaceNet.Segmentation(args, base_dir=args.dataset_basedir, split='val')
        return val_dataset
    elif args.dataset_name == 'LoveDA':
        val_dataset = data_LoveDA.Segmentation(args, base_dir=args.dataset_basedir, split='val')
        return val_dataset
