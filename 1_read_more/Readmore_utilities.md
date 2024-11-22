## Fundus utilities

A collection of additional utilities that can come in handy when working with fundus images.

- **ImageTorchUtils**: Image manipulation based on Pytorch tensors. <br>
    Example usage:
    ```python
    from fundus_image_toolbox.utilities import ImageTorchUtils as Img
    # Functions can be chained together. The final output is accessed via the `.img` attribute.
    # Load an image as an RGB tensor
    image = Img("path/to/image.jpg").to_tensor().img
    # Turn a cv2 image into an RGB tensor
    image = Img(cv2_image).to_tensor(from_cspace="bgr", to_cspace="rgb").img
    # Change color space of an RGB image and move the color channel to the last dimension
    image = Img(image).to_tensor().to_cspace(from_cspace="rgb", to_cspace="gray").set_channel_dim(-1).img
    # Get a numpy array from any RGB image or path, suited to be displayed with matplotlib
    image = Img(image).to_numpy().img
    # Move the color channel in an array to the last dimension
    image = Img(image).set_channel_dim(-1).img
    # Convert to pil image
    image = Img(image).to_tensor().to_pil().img
    # Get a torch tensor image batch from anything (path, cv2 image, PIL image, numpy array or lists, tuples or arrays of one type of these)
    image_batch = Img(image).to_batch().img
    # Check if you deal with a batch-like structur (list of images, tensor or tensors, etc.)
    is_batch = Img(image_batch).is_batch_like()
    # Check if you deal with a proper torch tensor batch
    is_batch = Img(image_batch).is_batch()
    ```
- **Balancing**: A script to balance a torch dataset by both oversampling the minority class(es) and undersampling the majority class(es) from [imbalanced-dataset-sampler](https://github.com/ufoym/imbalanced-dataset-sampler/). <br>
    Example usage to balance by a custom label. 
    ```python
    # In this example case, `data["dataset"]` refers to where the data originally came from before combining it into a single dataset
    from fundus_image_toolbox.utilities import ImbalancedDatasetSampler
    if config.balance_datasets:
        # Balance by custom label
        sampler = ImbalancedDatasetSampler(self.train_dataset, method="balanced", labels = self.train_dataset.data["dataset"])
        shuffle = False
    else:
        sampler = None
        shuffle = True
    # Create dataloader
    self.train_dataloader = torch.utils.data.DataLoader(
        self.train_dataset, batch_size=config.batch_size, shuffle=shuffle, sampler=sampler
        )
    ```

- **Fundus transforms**: A collection of torchvision data augmentation transforms to apply to fundus images adapted from [pytorch-classification](https://github.com/YijinHuang/pytorch-classification/blob/master/data/transforms.py). <br>
    Example usage:
    ```python
    from torchvision import transforms
    from fundus_image_toolbox.utilities import get_transforms, get_unnormalization
    # If mean and std are not provided, ImageNet values are used
    # If split is not "train", then only normalization, resize and centercrop are applied
    transformations = transforms.Compose(
        get_transforms(img_size = 512, split = "train", normalize=True, mean=MEAN, std=STD)
    )
    unnormalization = get_unnormalization(mean=MEAN, std=STD)
    ```

- **Get pixel mean std**: A script to calculate the mean and standard deviation of the pixel values by channel of a dataset from pytorch dataloaders. <br>
    Example usage:
    ```python
    from fundus_image_toolbox.utilities import get_pixel_mean_std
    # d: List of all torch dataloaders. Make sure not to augment the data in any loader.
    d = [train_loader, val_loader, test_loader] 
    mean, std = get_pixel_mean_std(d, device="cuda:0")
    ```
    
- **Model getter**: Getter for torchvision models with efficientnet and resnet architectures initialized with ImageNet weights. <br>
    Example usage:
    ```python
    from fundus_image_toolbox.utilities import get_efficientnet_or_resnet
    # Options: resnet{18,34,50,101,152}, efficientnet-b{0-7}
    model = get_efficientnet_or_resnet("resnet18", n_outs=1):
    ```

- **LR scheduler**: Get a pytorch learning rate scheduler (plus a warmup scheduler) for a given optimizer: Constant, OneCycleLR, CosineAnnealingLR, CosineAnnealingWarmRestarts. <br>
    Example usage:
    ```python
    from fundus_image_toolbox.utilities import get_lr_scheduler
    # Options: "cosine", "cosine_restart", "onecycle"
    # Without warmup scheduler:
    self.scheduler = get_lr_scheduler(
        "onecycle", optimizer, train_loader, lr=1e-4, epochs=25, cosine_len=None, warmup_epochs=None
        )

    # With warmup scheduler:
    self.scheduler, self.warmup_scheduler = get_lr_scheduler(
        "onecycle", optimizer, train_loader, lr=1e-4, epochs=25, cosine_len=None, warmup_epochs=5
        )

    # Use like this in your script:
    #   Here, the scheduler string (e.g. "onecycle") is stored in self.cfg.scheduler.
    #   In the training loop, at the beginning of each epoch, run:

    if self.warmup_scheduler and not self.warmup_scheduler.finished():
            self.warmup_scheduler.step()

    #   And after each batch, run:
    if self.scheduler and "onecycle" in self.cfg.scheduler:
        if not self.warmup_scheduler or self.warmup_scheduler.finished() and self.warmup_scheduler.epoch != self.current_epoch+1:
            self.scheduler.step()

    #   And at the end of each epoch, run:
    if self.scheduler and not "onecycle" in self.cfg.scheduler:
        if not self.warmup_scheduler or self.warmup_scheduler.finished():
            self.scheduler.step()
    ```

- **Multilevel 3-way split**: Split a pandas dataframe into train, validation and test splits with the options to split by group (i.e. keep groups together) and stratify by label. Wrapper for [multi_level_split](https://github.com/lmkoch/multi-level-split/). <br>
    Example usage:
    ```python
    from fundus_image_toolbox.utilities import multilevel_3way_split, multilevel_train_test_split
    # Fix this for each of your experiments to guarantee equal splits across runs. If None, 12345 is used.
    SPLIT_SEED = 12345 
    train_df, val_df, test_df = multilevel_3way_split(
        df, proportions=[0.6, 0.2, 0.2], seed=SPLIT_SEED, stratify_by="dr_grade", split_by="patient_id"
    )

    # Alternatively, split only into train and test using lmkoch's split function:
    train_df, test_df = multilevel_train_test_split(
        df, index=df.index, test_split=0.2, seed=SPLIT_SEED, stratify_by="dr_grade", split_by="patient_id"
    )
    ```

- **Seed everything**: Set seed for reproducibility in python, numpy and torch, depending on the availability of the respective libraries. <br>
    Example usage:
    ```python
    from fundus_image_toolbox.utilities import seed_everything
    seed_everything(SEED)
    ```

- **Other basics**: 
    ```python
    from fundus_image_toolbox.utilities import exist, exists, flatten_one, parse_list, on_slurm_job
    
    # Check if a file or multiple files exist
    exists("path/to/file") # > True
    exist(["path/to/file1", "path/to/file2"]) # > True
    
    # Flatten by one level
    flatten_one([[1,2], [3,[4,5]]]) # > [1, 2, 3, [4, 5]]
    
    # Parse list-like structures as a list
    parse_list("[1,2,3]") # > [1, 2, 3]

    # Check if running on a slurm job
    on_slurm_job() # > True
    ```

