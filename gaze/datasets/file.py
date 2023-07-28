import os
from GazeFollow import GazeFollow

if __name__ == '__main__':
        labels = os.path.join('datasets/gazefollow_extended', "train_annotations_release.txt" )
        # root_dir datasets/gazefollow_extended
        # labels they are coming from file 
        dataset = GazeFollow('datasets/gazefollow_extended', labels, input_size=224, output_size=64, is_test_set=False)
        from torch.utils.data import DataLoader

        loader = DataLoader(
            dataset=dataset, batch_size=16, shuffle=True, num_workers=6, pin_memory=True
        )
        print(type(loader))