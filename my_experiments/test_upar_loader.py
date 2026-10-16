import os

from upar_dataset.UPAR.UPAR import UPAR, get_transform

DATA_PATH = os.path.join('d:/AI Assignment', 'upar_dataset', 'UPAR', 'dataset_all.pkl')
ROOT_PATH = os.path.join('d:/AI Assignment', 'datasets')


def test_upar_loader_initializes():
    train_transform, valid_transform = get_transform(256, 128)
    dataset = UPAR(
        split='train',
        partition=0,
        root=ROOT_PATH,
        data_path=DATA_PATH,
        transform=train_transform,
        target_transform=None,
    )

    assert len(dataset) > 0
    assert dataset.attr_num == 40
    assert dataset.img_idx.shape[0] == len(dataset)
    assert dataset.label.shape == (len(dataset), 40)
