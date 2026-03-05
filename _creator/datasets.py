from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from pathlib import Path

# all datasets here should:
# - have an argument transform for the input transformation (can be ignored if you want)
# - have their underlying dataframe available at self.df
# - have their underlying path column available at self.path_col, have an img_root member (None if unused)
# - have their underlying label column available at self.label_col (must be str)

class ImageDataset(Dataset):
    """A class to use images as a dataset.

    df_path:            Path to csv file containing paths and labels for the images
    path_col_name:      Name of the column containing the image paths
    label_col_name:     Name of the column containing the labels
    cat_col_names:      List of column names of categorical columns (as defined in extra_data_columns)
    class_names:        List of class names at their logit position.
    transform:          torchvision.transforms transformation. do not include directly in the dataset args in the config file!
    filter_col:         dict(col_name --> [valid_value0, valid_value1, ...]) indicating column names and valid values for these
                        columns. the condition for the different columns will be ANDed, the conditions for the different
                        values will be ORed.
                        --> allows for filtering the dataframe for rows where specific column values occur.
    img_root:           Root directory for the dataframe's path column. If None, it is assumed the paths are already absolute.
    """
    def __init__(self, df_path, path_col_name, label_col_name, cat_col_names, class_names, transform, filter_cols=None,
                 img_root=None):
        super(ImageDataset, self).__init__()
        self.df = pd.read_csv(df_path).astype({label_col_name: str, **{name: str for name in cat_col_names}})
        if filter_cols is not None:
            for col_name, valid_values in filter_cols.items():
                self.df = self.df[self.df[col_name].isin(valid_values)].reset_index() # keep all columns for optionally tracking other labels later on
        self.label_col = self.df[label_col_name]
        self.path_col = self.df[path_col_name]
        self.tfms = transform
        self.class_dict = {name: i for i, name in enumerate(class_names)}
        self.img_root = img_root
        if self.img_root is not None:
            self.img_root = Path(img_root)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        p = self.path_col[item]
        if self.img_root is not None:
            p = str(self.img_root/p)
        img = Image.open(p)
        img = self.tfms(img.convert("RGB"))
        label = self.class_dict[self.label_col[item]]
        return img, label


# internal use only, don't use this in any configs
class _ImageFolderDataset(Dataset):
    def __init__(self, img_folder, transform):
        super(_ImageFolderDataset, self).__init__()
        self.tfms = transform
        self.img_folder = img_folder
        self.img_paths = sorted([
            p for p in self.img_folder.iterdir() if p.suffix.lower() in [
                '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'
            ]], key=lambda p: tuple(map(int, p.stem.split("_"))) # sort primarily by y-coordinate, then x-coordinate
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        p = self.img_paths[item]
        img = Image.open(p)
        img = self.tfms(img.convert("RGB"))
        return img, -1 # dummy label to keep output format consistent with ImageDataset
