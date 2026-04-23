
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
import numpy as np
from typing import Union

class ImageTorchUtils:
    """Class for image manipulation based on Pytorch. Represents an image as a torch tensor of 
    shape (C, H, W), even for uncolored images (there, using an empty channel dimension).
    
    Usage:
    ```python
    import ImageTorchUtils as Img
    # Use for single image manipulation
    # Pass a path to an image or an image as a numpy array, PIL Image or torch tensor
    # and chain the desired methods
    img_path = "path/to/image.jpg"
    img = Img(img_path).to_tensor().img # > torch.Tensor
    img = Img(img_path).to_tensor().to_cspace(from_cspace="rgb", to_cspace="bgr").img # > torch.Tensor in BGR

    # Alternatively, use it to get a batch of images from a list of images or a list of 
    # paths to images
    img_paths = ["path/to/image1.jpg", "path/to/image2.jpg"]
    img_batch = Img(img_paths).to_batch().img

    # If your image is uncolored (greyscale or mask), use the img_ndims argument for the
    # `to_tensor` and `to_batch` functions:
    img_2d_path = "path/to/mask.png" # -> (H, W)
    img_2d = Img(img_2d_path).to_tensor(img_ndims=2).img # > torch.Tensor of shape (1, H, W)
    img_2d_batch = Img(img_2d_path).to_batch(img_ndims=2).img # > torch.Tensor of shape (B, 1, H, W)
    ```
    """
    def __init__(self, image: Union[str, Image.Image, np.ndarray, torch.Tensor, list]):
        self.img = image

    def to_tensor(
        self,
        from_cspace: str = 'rgb',
        to_cspace: str = 'rgb',
        silent: bool = False,
        img_ndims: int = 3,
    ):
        """Converts the instance's image to a torch tensor of shape (C, H, W).

        Args:
            from_cspace (str, optional): Colorspace of the image. Defaults to 'rgb'.
            to_cspace (str, optional): Colorspace to convert the image to. Defaults to 'rgb'.
            img_ndims (int, optional): Expected dimensions of one image (3 for
                color images (default) and 2 for grayscale/mask images). Is used to un-ambiguously 
                differentiate between a single image and a batch of images.
        Returns:
            torch.Tensor: Image as a torch tensor.
        """
        if img_ndims not in [2, 3]:
            raise ValueError("img_ndims should be 2 or 3 but is", img_ndims)
        
        is_batch = False
        is_batch_like = False
        if img_ndims == 2:
            if np.squeeze(np.asarray(self.img)).ndim == 2 and np.asarray(self.img).ndim >= 3:
                self.img = self.squeeze(dim=0).img
            is_batch = self._is_batch_2d()
            is_batch_like = self._is_batch_2d_like()
        elif img_ndims == 3:
            is_batch = self.is_batch()
            is_batch_like = self.is_batch_like()

        # Case: Batch of images
        if is_batch:
            if None not in [from_cspace, to_cspace] and not silent:
                print("You passed a batch to to_tensor. In this case, from_cspace and to_cspace are ignored. Use to_cspace on the batch elements instead.")
            return self
        if is_batch_like:
            if None not in [from_cspace, to_cspace] and not silent:
                print("You passed a batch-like object to to_tensor. In this case, from_cspace and to_cspace are ignored for batch images. Use to_cspace instead.")
            imgs = [ImageTorchUtils(img).to_tensor(img_ndims=img_ndims).img for img in self.img]
            self.img = torch.stack(imgs)
            return self
        
        # Case: Single image: Convert to tensor of shape (C, H, W), even for uncolored images (there, using C=1)
        from_cspace = from_cspace.lower().replace(' ', '')
        from_cspace = "gray" if from_cspace in ['gray', 'grey'] else from_cspace
        if from_cspace not in ['rgb', 'bgr', 'gray', 'grey']:
            raise ValueError("from_cspace should be one of ['rgb', 'bgr', 'gray'] but is", from_cspace)
        
        if isinstance(self.img, str):
            # PIL Image with shape (H, W) or (H, W, C) -> (C, H, W) (to_tensor adds channel dim if absent)
            self.img = to_tensor(Image.open(self.img)) 
            from_cspace = 'rgb'
        elif isinstance(self.img, torch.Tensor):
            if len(self.img.shape) == 2:
                # Grayscale torch tensor with shape (H, W) -> (1, H, W)
                self.img = self.img.unsqueeze(0)
        elif isinstance(self.img, Image.Image):
            # PIL Image with shape (H, W) or (H, W, C) -> (C, H, W) (to_tensor adds channel dim if absent)
            self.img = to_tensor(self.img)
        elif isinstance(self.img, np.ndarray):
            # Greyscale  numpy image
            if len(self.img.shape) == 2:
                # Grayscale numpy array with shape (H, W) -> (1, H, W)
                self.img = np.expand_dims(self.img, 2) 
                self.img = to_tensor(self.img) 
            elif len(self.img.shape) == 3 and self.img.shape[2] == 1:
                # Grayscale numpy array with shape (H, W, 1) -> (1, H, W)
                self.img = to_tensor(self.img)
            elif len(self.img.shape) == 3 and self.img.shape[0] == 1:
                # Grayscale numpy array with shape (1, H, W) -> (1, H, W)
                # Here, to_tensor(Image.fromarray(self.img)) would raise error
                self.img = torch.as_tensor(self.img)
            else:
                # Color (RGB/BGR) numpy array
                if np.issubdtype(self.img.dtype, np.floating):
                    self.img = (self.img * 255).astype(np.uint8)
                if len(self.img.shape) == 3 and self.img.shape[0] == 3:
                    # We have (C, H, W) but Image.fromarray expects (H, W, C)
                    self.img = self.img.transpose(1, 2, 0) # -> (H, W, C)
                # (H, W, C) -> (C, H, W)
                self.img = to_tensor(Image.fromarray(self.img))
        else:
            raise ValueError("Image should be a string, numpy array (as from plt.imread or cv2.imread), PIL Image or torch tensor but is of type", type(self.img))
        
        self.img = self.to_cspace(from_cspace, to_cspace).img

        return self
    
    def squeeze(self, dim: int = 0):
        """Squeezes the instance's image tensor along the specified dimension.

        Args:
            dim (int, optional): Dimension to squeeze. Defaults to 0.

        Returns:
            torch.Tensor: Image tensor with the specified dimension squeezed.
        """
        if self.is_batch_like():
            # Recursively squeeze each image in the batch
            imgs = [ImageTorchUtils(img).squeeze(dim).img for img in self.img]
            return self

        if self.is_batch():
            # Recursively squeeze each image in the batch
            imgs = [ImageTorchUtils(img).squeeze(dim).img for img in self.img]
            self.img = torch.stack(imgs)
            return self
        
        if isinstance(self.img, torch.Tensor) and self.img.shape[dim] == 1: #len(self.img.shape) == 4 and 
            self.img = self.img.squeeze(dim)
        elif isinstance(self.img, np.ndarray) and self.img.shape[dim] == 1: # and len(self.img.shape) == 4
            self.img = np.squeeze(self.img, dim)
        elif isinstance(self.img, Image.Image) and np.asarray(self.img).shape[dim] == 1: # and len(np.array(self.img).shape) == 4 
            self.img = np.asarray(self.img).squeeze(dim)
            self.img = Image.fromarray(self.img)
        else:
            if not isinstance(self.img, (torch.Tensor, np.ndarray, Image.Image)):
                raise ValueError("Image should be a torch tensor, Image.Image or numpy array but is of type", type(self.img))        
        
        return self
    
    def to_cspace(self, from_cspace: str, to_cspace: str):
        """Converts the instance's image tensor from one colorspace to another.

        Args:
            from_cspace (str): Colorspace of the image.
            to_cspace (str): Colorspace to convert the image to.

        Returns:
            torch.Tensor: Image in the new colorspace.
        """
        if not isinstance(self.img, torch.Tensor):
            raise ValueError("Image should be a torch tensor but is of type", type(self.img))

        if self.is_batch():
            # Recursively convert each image in the batch
            imgs = [ImageTorchUtils(img).to_cspace(from_cspace, to_cspace).img for img in self.img]
            self.img = torch.stack(imgs)
            return self

        from_cspace = from_cspace.lower().replace(' ', '')
        to_cspace = to_cspace.lower().replace(' ', '')
        from_cspace = "gray" if from_cspace in ['gray', 'grey'] else from_cspace
        to_cspace = "gray" if to_cspace in ['gray', 'grey'] else to_cspace
        if from_cspace not in ['rgb', 'bgr', 'gray', 'grey']:
            raise ValueError("from_cspace should be one of ['rgb', 'bgr', 'gray'] but is", from_cspace)
        if to_cspace not in ['rgb', 'bgr', 'gray', 'grey']:
            raise ValueError("to_cspace should be one of ['rgb', 'bgr', 'gray'] but is", to_cspace)
        
        if from_cspace == 'gray':
            self.img = self.img.expand(3, -1, -1)      
        elif from_cspace == 'bgr':
            self.img = self.img.flip(0)
        elif from_cspace == 'rgb':
            pass
            
        if to_cspace == 'gray':
            self.img = self.img.mean(0, keepdim=True)
        elif to_cspace == 'bgr':
            self.img = self.img.flip(0)
        elif to_cspace == 'rgb':
            pass
        return self
    
    def set_channel_dim(self, channel_dim: int):
        """Sets the channel dimension of the instance's image tensor.

        Args:
            dim (int): The desired channel dimension.
        """       
        if self.is_batch_like():
            # Recursively set the channel dimension for each image in the batch
            input_type = type(self.img[0])
            imgs = [ImageTorchUtils(img).set_channel_dim(channel_dim).img for img in self.img]
            if input_type == np.ndarray:
                imgs = np.asarray([np.asarray(img) for img in imgs])
            else:
                self.img = torch.stack(imgs)
            return self

        input_type = type(self.img)

        if not isinstance(self.img, torch.Tensor):
            self.img = self.to_tensor().img

        if len(self.img.shape) != 3:
            raise ValueError("Image should have 3 dimensions but has", len(self.img.shape))
        
        # Index of the current channel dimension: Smallest shape
        current_channel_dim = self.img.shape.index(min(self.img.shape))

        if channel_dim == -1:
            channel_dim = len(self.img.shape) - 1
        if channel_dim >= len(self.img.shape):
            raise ValueError("channel_dim should be less than the number of dimensions of the image tensor but is", channel_dim)
        if channel_dim < 0:
            raise ValueError("channel_dim should be -1 or greater than or equal to 0 but is", channel_dim)

        if current_channel_dim != channel_dim:
            newdims = list(range(len(self.img.shape)))
            newdims.remove(current_channel_dim)
            newdims.insert(channel_dim, current_channel_dim)
            self.img = self.img.permute(*newdims)

        if input_type == np.ndarray:
            self.img = self.img.numpy()
        
        return self

    def to_pil(self):
        """Converts the instance's image tensor to a PIL Image.

        Returns:
            Image: The image as a PIL Image.
        """
        if not isinstance(self.img, torch.Tensor):
            raise ValueError("Image should be a torch tensor but is of type", type(self.img))
        
        if self.is_batch():
            # Recursively convert each image in the batch. Returning a list of PIL Images.
            imgs = [ImageTorchUtils(img).to_pil().img for img in self.img]
            self.img = imgs
            return self
        
        self.img = to_pil_image(self.img)
        return self
    
    def to_numpy(self, dtype: str="uint8"):
        """Converts the instance's image tensor to a numpy array (H, W, C).

        Args:
            image (torch.Tensor): Image tensor.
            dtype (str, optional): Data type of the numpy array. Defaults to "uint8".
                Choices: "uint8", "float32".

        Returns:
            np.ndarray: Numpy array.
        """
        if isinstance(self.img, np.ndarray) and self.img.dtype == dtype:
            return self
        
        if self.is_batch_like():
            # Recursively convert each image in the batch. Returning an narray of numpy arrays.
            imgs = [ImageTorchUtils(img).to_numpy(dtype).img for img in self.img]
            self.img = np.asarray(imgs)
            return self
        
        # if not isinstance(self.img, torch.Tensor):
        # (H, W, C) as is standard for numpy arrays
        self.img = self.to_tensor().set_channel_dim(-1).img 

        if dtype == "uint8":
            if self.img.max() <= 1:
                self.img = (self.img * 255).byte()
            else:
                self.img = self.img.byte()
        elif dtype == "float32":
            self.img = self.img.float()
        else:
            raise ValueError("dtype should be one of ['uint8', 'float32'] but is", dtype)
        
        self.img = self.img.numpy()
        return self
    
    def to_uint8(self):
        """Converts the instance's image tensor to a numpy array with dtype uint8.

        Returns:
            np.ndarray: The image as a numpy array with dtype uint8.
        """
        if isinstance(self.img, np.ndarray) and self.img.dtype == np.uint8:
            return self

        if self.is_batch_like():
            # Recursively convert each image in the batch
            imgs = [ImageTorchUtils(img).to_uint8().img for img in self.img]
            self.img = np.asarray(imgs)
            return self

        if not isinstance(self.img, np.ndarray):
            self.img = self.to_numpy().img
        
        if self.img.dtype == np.float32:
            self.img = (self.img * 255).astype(np.uint8)
        return self
    
    def to_float32(self):
        """Converts the instance's image tensor to a numpy array with dtype float32.

        Returns:
            np.ndarray: The image as a numpy array with dtype float32.
        """
        if isinstance(self.img, np.ndarray) and self.img.dtype == np.float32:  
            return self

        if self.is_batch_like():
            # Recursively convert each image in the batch
            imgs = [ImageTorchUtils(img).to_float32().img for img in self.img]
            self.img = np.asarray(imgs)
            return self

        if not isinstance(self.img, np.ndarray):
            self.img = self.to_numpy().img
        
        if self.img.dtype == np.uint8:
            self.img = self.img.astype(np.float32) / 255
        return self
    
    def to_batch(self, img_ndims: int = 3):
        """Converts the instance's img object to a tensor batch of images (shape (B, C, H, W)).
        The instance's img object can be a list, string, numpy array, PIL Image or torch tensor corresponding to
        a single image or a list of the same or a path or list of paths to images.

        Args:
            img_ndims (int, optional): Expected dimensions of one image (3 for
                color images (default) and 2 for grayscale/mask images). Is used to un-ambiguously 
                differentiate between a single image and a batch of images.

        Returns:
            torch.Tensor: Batch of images.
        """
        if img_ndims not in [2, 3]:
            raise ValueError("img_ndims should be 2 or 3 but is", img_ndims)
        
        is_batch = False
        is_batch_like = False
        if img_ndims == 2:
            is_batch = self._is_batch_2d()
            is_batch_like = self._is_batch_2d_like()
        elif img_ndims == 3:
            is_batch = self.is_batch()
            is_batch_like = self.is_batch_like()

        # Case: Batch of images
        if is_batch:
            return self
        
        # Case: Single image 
        # Convert single image to list
        if not is_batch_like:
            self.img = [self.img]
        
        # Paths, nd.arrays, PIL Images to image tensors of shape (C, H, W); After to_tensor, each image is of shape (C, H, W), never (H, W)
        if isinstance(self.img[0], (str, Image.Image, np.ndarray, torch.Tensor)):
            self.img = [ImageTorchUtils(image).to_tensor(img_ndims=img_ndims).img for image in self.img]
        else:
            raise ValueError("Image should be a list of paths, numpy arrays, PIL Images or torch tensors but is of type", type(self.img[0]))
        
        # Find the smallest image size
        min_height = min(img.shape[1] for img in self.img)
        min_width = min(img.shape[2] for img in self.img)
        
        # Crop images to the smallest size to prevent error in torch.stack
        self.img = [img[:, :min_height, :min_width] for img in self.img]
        
        self.img = torch.stack(self.img)
        return self

    def is_batch(self) -> bool:
        """Checks if the instance's image is a tensor batch of tensor images.
        Use to_batch to convert lists or arrays of images or a single image to a batch of images.

        Returns:
            bool: True if the image is a batch of images, False otherwise.
        """
        return isinstance(self.img, torch.Tensor) and len(self.img.shape) == 4 and [isinstance(img, torch.Tensor) for img in self.img]

    def _is_batch_2d(self) -> bool:
        """Internal function to check if the instance's image is a tensor batch of uncolored images.

        Returns:
            bool: True if the image is a batch of uncolored images, False otherwise.
        """
        return isinstance(self.img, torch.Tensor) and len(self.img.shape) == 3 and [isinstance(img, torch.Tensor) for img in self.img]
    
    def is_batch_like(self) -> bool:
        """Checks if the instance's image is a list or array of images or a single image.

        Returns:
            bool: True if the image is a list or array of images, False otherwise.
        """
        return self.is_batch() or isinstance(self.img, (list, tuple)) or len(np.asarray(self.img).shape) == 4

    def _is_batch_2d_like(self) -> bool:
        """Internal function to check if the instance's image is a list or array of uncolored images.

        Returns:
            bool: True if the image is a list or array of uncolored images, False otherwise.
        """
        return self._is_batch_2d() or isinstance(self.img, (list, tuple)) or len(np.asarray(self.img).shape) == 3
