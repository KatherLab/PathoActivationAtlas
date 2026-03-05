import time
import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from torchvision.transforms import v2
from pathlib import Path
import argparse
import timm
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
)
import PIL
from tabulate import tabulate

# transform = transforms.Compose(
#     [
#         transforms.Resize(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#     ]

torch.set_float32_matmul_precision("high")

class LitNetWorkUni(L.LightningModule):
    def __init__(self, num_classes, model_dir,model_freeze=False, optimizer="adamW", lr=0.001):
        super().__init__()

        self.model = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        self.model.load_state_dict(torch.load(os.path.join(model_dir, "uni.bin"), map_location="cpu", weights_only=True), strict=True)

        self.base_model = "Uni"

        self.model_freeze = model_freeze
        self.optimizer_name = optimizer
        self.lr = lr

        if model_freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.head = torch.nn.Linear(
            in_features=1024, out_features=num_classes, bias=True
        )

        for param in self.model.head.parameters():
            param.requires_grad = True

        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.valid_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_auroc = MulticlassAUROC(num_classes=num_classes, average="macro")
        self.test_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes)

        if self.optimizer_name == "adamW":
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        elif self.optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        else:
            raise NotImplementedError(f"Optimizer {self.optimizer_name} has not been implemented.")

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x, y = batch

        preds = self.model(x)

        if not isinstance(preds, torch.Tensor):
            preds = preds["pooler_output"]

        loss = torch.nn.functional.cross_entropy(preds, y)

        self.train_acc(preds, y)

        self.log(
            "train_acc", self.train_acc, on_step=True, on_epoch=False, prog_bar=True
        )
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch

        preds = self.model(X)

        if not isinstance(preds, torch.Tensor):
            preds = preds["pooler_output"]

        loss = torch.nn.functional.cross_entropy(preds, y)

        self.valid_acc(preds, y)

        self.log(
            "valid_acc", self.valid_acc, on_step=True, on_epoch=True, prog_bar=True
        )
        self.log("valid_loss", loss, on_epoch=True, on_step=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        X, y = batch

        preds = self.model(X)

        if not isinstance(preds, torch.Tensor):
            preds = preds["pooler_output"]

        loss = torch.nn.functional.cross_entropy(preds, y)

        self.test_acc(preds, y)
        self.test_auroc(preds, y)
        self.test_f1(preds, y)
        self.confusion_matrix(preds, y)

        self.log("test_acc", self.test_acc, on_step=True, on_epoch=True)
        self.log("test_loss", loss, on_step=True, on_epoch=True)
        self.log("test_auroc", self.test_auroc, on_step=True, on_epoch=True)
        self.log("test_f1", self.test_f1, on_step=True, on_epoch=True)

    # def on_test_epoch_end(self):
    #     confusion_matrix_computed = (
    #         self.confusion_matrix.compute().detach().cpu().numpy().astype(int)
    #     )

    #     df_cm = pd.DataFrame(confusion_matrix_computed)
    #     plt.figure(figsize=(10, 7))
    #     fig_ = sns.heatmap(df_cm, annot=True, cmap="Spectral").get_figure()
    #     plt.close(fig_)
    #     self.loggers[0].experiment.add_figure(
    #         "Confusion matrix", fig_, self.current_epoch
    #     )

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        # optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        return self.optimizer

class PathologyDataset(Dataset):
    def __init__(self, csv_file, base_path=None,transforms=None):
        self.data = pd.read_csv(csv_file)
        self.transforms = transforms
        self.base_path = base_path
        
        # Convert labels to numeric if they're strings
        if self.data['label'].dtype == 'object':
            # Sort unique labels to ensure consistent mapping
            unique_labels = sorted(self.data['label'].unique())
            self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
            self.reverse_mapping = {idx: label for label, idx in self.label_mapping.items()}
            self.data['label_encoded'] = self.data['label'].map(self.label_mapping)
        else:
            # If labels are already numeric, create identity mapping
            unique_labels = sorted(self.data['label'].unique())
            self.label_mapping = {label: int(label) for label in unique_labels}
            self.reverse_mapping = {int(label): label for label in unique_labels}
            self.data['label_encoded'] = self.data['label']
        
        # Print label distribution
        self.label_distribution = self.data['label'].value_counts().sort_index()
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row['path']
        label = row['label_encoded']
        
        # Load and convert image
        if self.base_path:
            image_path = os.path.join(self.base_path, image_path)
        image = PIL.Image.open(image_path).convert('RGB')
        
        if self.transforms:
            image = self.transforms(image)
            
        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.long)
            
        return image, label
    
    def print_label_info(self):
        """Print detailed information about labels and their mapping."""
        print("\nLabel Mapping (Class Index → Label Name):")
        mapping_table = [[idx, label, self.label_distribution[label]] 
                        for idx, label in self.reverse_mapping.items()]
        print(tabulate(mapping_table, 
                      headers=['Class Index', 'Label Name', 'Count'], 
                      tablefmt='grid'))
        print(f"\nTotal samples: {len(self.data)}\n")
        return mapping_table

def get_transforms(is_training=True):
    if is_training:
        return v2.Compose([
            v2.RandomResizedCrop(size=(224, 224), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=90),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return v2.Compose([
            v2.Resize(256, antialias=True),
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])



def train(args):
    # Load split configuration
    splits_dir = Path(args.splits_dir)
    with open(splits_dir / "split_config.json", "r") as f:
        split_config = json.load(f)
    
    # Setup transforms
    # transforms = v2.Compose([
    #     v2.ToImage(),
    #     v2.RandomResizedCrop(size=(224, 224), antialias=True),
    #     v2.RandomHorizontalFlip(p=0.1),
    #     v2.ToDtype(torch.float32, scale=True),
    #     v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
    
    # Load test dataset
    test_dataset = PathologyDataset(
        splits_dir / "test_split.csv",
        base_path=args.base_path,
        transforms=get_transforms(is_training=False)
    )

    print(f"\n=== Test Set Label Information ===")
    _ = test_dataset.print_label_info()

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Training loop for each fold
    for fold in range(split_config['n_splits']):

        if args.folds is not None and fold not in args.folds:
            print(f"Skipping fold {fold}, not specified in --folds argument")
            continue

        print(f"Training Fold {fold}")
        fold_dir = splits_dir / f"fold_{fold}"
        
        # Create results directory for this fold
        results_dir = Path(args.results_dir) / Path(f"{args.experiment_name}_{time.strftime('%Y%m%d_%H%M%S')}")  / f"fold_{fold}"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load train and validation datasets
        train_dataset = PathologyDataset(
            fold_dir / "train.csv",
            base_path=args.base_path,
            transforms=get_transforms(is_training=True)
        )
        val_dataset = PathologyDataset(
            fold_dir / "val.csv",
            base_path=args.base_path,
            transforms=get_transforms(is_training=False)
        )

        # Print label information for train and validation sets
        print(f"\n=== Fold {fold} Training Set Label Information ===")
        _ = train_dataset.print_label_info()
        print(f"\n=== Fold {fold} Validation Set Label Information ===")
        _ = val_dataset.print_label_info()
        
        # Save label mapping and distribution
        mapping_info = {
            'label_mapping': train_dataset.label_mapping,
            'reverse_mapping': train_dataset.reverse_mapping,
            'label_distribution': train_dataset.label_distribution.to_dict()
        }
        
        with open(results_dir / "label_info.json", 'w') as f:
            json.dump(mapping_info, f, indent=4)
            
        # Verify label consistency across splits
        if not (train_dataset.label_mapping == val_dataset.label_mapping == test_dataset.label_mapping):
            raise ValueError("Label mappings are not consistent across datasets!")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        # Initialize model
        if args.base_model == "uni":
            model = LitNetWorkUni(
                num_classes=args.num_classes,
                model_freeze=args.model_freeze,
                optimizer=args.optimizer,
                lr=args.lr,
                model_dir=args.model_dir
            )
        elif args.base_model == "phicon":
            # model = LitNetWorkPhicon(
            #     num_classes=args.num_classes,
            #     model_freeze=args.model_freeze,
            #     optimizer=args.optimizer,
            #     lr=args.lr
            # )
            raise NotImplementedError(f"Model not implemented: {args.base_model}")
        elif args.base_model == "ctranspath" or args.base_model == "ctp":
            # model = LitNetWorkViT(
            #     num_classes=args.num_classes,
            #     model_freeze=args.model_freeze,
            #     optimizer=args.optimizer,
            #     lr=args.lr,
            #     model_name=args.base_model
            # )
            raise NotImplementedError(f"Model not implemented: {args.base_model}")
        else:
            raise NotImplementedError(f"Model not implemented: {args.base_model}")
        
        # Setup trainer
        trainer = L.Trainer(
            max_epochs=args.max_epochs,
            callbacks=[
                EarlyStopping(
                    monitor="valid_loss",
                    mode="min",
                    patience=args.patience
                ),
                ModelCheckpoint(
                    dirpath=results_dir / "checkpoints",
                    filename="best-checkpoint",
                    monitor="valid_loss",
                    mode="min"
                ),
                LearningRateMonitor(logging_interval="step")
            ],
            default_root_dir=results_dir,
            accelerator=args.accelerator,
            devices=1,
            precision=args.precision
        )
        
        # Train and evaluate
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )
        
        # Test
        trainer.test(model=model, dataloaders=test_loader)

def parse_folds(value):
    # Split the input string by commas and convert each part to an integer
    return [int(x) for x in value.split(',')]


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training script using pre-generated splits")
    
    # Data arguments
    parser.add_argument("--model_dir", type=str, required=True,
                      help="Directory containing the base models (foundation models)")
    parser.add_argument("--splits_dir", type=str, required=True,
                      help="Directory containing the generated splits")
    parser.add_argument("--results_dir", type=str, required=True,
                      help="Directory to save results")
    parser.add_argument("--base_path", type=str, default=None,
                      help="Base path to the dataset. If empty, assume correct relative / absolute paths in the CSV files.")
    parser.add_argument("--experiment_name", type=str, required=True,
                      help="Name of the experiment")

    # select folds to train (optional, list with int comma-separated)
    parser.add_argument("--folds", type=parse_folds, default=None, help="Folds to train on. If not specified, train on all folds")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--precision", type=str, default="32",
                      choices=["32", "16", "bf16"])
    parser.add_argument("--accelerator", type=str, default="auto", choices=["auto", "cpu", "gpu"])
    
    # Model parameters
    parser.add_argument("--num_classes", type=int, default=32)
    parser.add_argument("--optimizer", type=str, default="adamW",
                      choices=["adamW", "sgd", "adam"])
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--model_freeze", action="store_true")
    parser.add_argument("--base_model", type=str, default="uni",
                      choices=["uni", "phicon", "ctranspath", "ctp"])

    

    
    args = parser.parse_args()
    
    train(args)