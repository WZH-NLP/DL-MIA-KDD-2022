from datasets import dataset_factory
from .bert import BertDataloader
from .ae import AEDataloader


DATALOADERS = {
    BertDataloader.code(): BertDataloader,
    AEDataloader.code(): AEDataloader
}


def dataloader_factory(args):
    dataset = dataset_factory(args)
    dataloader = DATALOADERS[args.dataloader_code]  # dataloader = BertDataloader
    dataloader = dataloader(args, dataset)  # is equals to dataloader = BertDataloader(args, dataset)
    train, val, test, usermap, itemmap = dataloader.get_pytorch_dataloaders()
    return train, val, test, usermap, itemmap
