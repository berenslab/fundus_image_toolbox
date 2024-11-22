## A Fovea and optic disc localization model for fundus images
A model to predict the center coordinates of the fovea and the optic disc in fundus images based on a multi-task EfficientNet trained on ADAM, REFUGE and IDRID datasets.

<img src="../fovea_od_localization/ex1.png" alt="Example image" width="800"/>
<br>Example predictions from the external dataset "DeepDRiD".

<br>

[x] Works on tensor images <br>
[x] Has batch support

### Preparation
- If you want to use the model, no preparation is needed and you can skip this part. Weights will be aquired automatically from zenodo.
- If you want to train or evaluate a model, you have to download these three datasets and store the extracted datasets into a common parent folder s.t. it contains the subdirectories `ADAM`, `REFUGE` and `IDRID`.
    - [ADAM dataset](https://doi.org/10.48550/arXiv.2202.07983) from [baidu](https://ai.baidu.com/broad/download) (paper: [link](https://doi.org/10.1109/TMI.2022.3172773))
    - [REFUGE dataset](https://doi.org/10.48550/arXiv.1910.03667) from [baidu](https://ai.baidu.com/broad/download) (paper: [link](https://doi.org/10.1016/j.media.2019.101570))
    - [IDRID dataset](https://doi.org/10.1016/j.media.2019.101561) from [ieee](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) (paper: [link](https://doi.org/10.1016/j.media.2019.101561))

- As the ADAM and REFUGE datasets provide optic disc masks instead of center coordinates, we have to extract the center coordinates from the masks. This is done in the [misc_masks_to_coordinates_ADAM.ipynb](../0_example_usage/fovea_od data preparation/misc_fovea-od_masks_to_coordinates_ADAM.ipynb) and [misc_masks_to_coordinates_REFUGE.ipynb](../0_example_usage/fovea_od data preparation/misc_masks_to_coordinates_REFUGE.ipynb) scripts. Finally, combine all datasets using the [misc_combine_IDRID_ADAM_REFUGE.ipynb](../0_example_usage/fovea_od data preparation/misc_combine_IDRID_ADAM_REFUGE.ipynb) script.

### How to
- See [usage_fovea-od_inference.ipynb](../0_example_usage/usage_fovea-od_inference.ipynb) on how to use the model.
- See [training_fovea-od.ipynb](../0_example_usage/training and evaluation/training_fovea-od.ipynb) for training a model from scratch.
- See [training_fovea-cli.ipynb](../0_example_usage/training and evaluation/training_fovea-od_cli.py) for training a model from the command line:
    You can pass a config file
    ```bash
    python training_fovea-od_cli.py --config /path/to/config.yaml
    ```
    Or, instead, set the respective config entries via command line arguments, see the help:
    ```bash
    python training_fovea-od_cli.py --help
    ```

### Performance
- On the test set of the combined dataset, the model achieves a mean distance to the fovea and optic disc targets of 0.88 % of the image size. This corresponds to a distance of 3,08 pixels in the 350 x 350 pixel images used for training and testing.
- Comparison of the above value to the winning model of the ADAM challenge wrt. the fovea localization task: The reported mean distance to the fovea target is 18.5538 pixels. This corresponds to a relative distance of 0.98 % of the image size*. Note that we did not evaluate on the ADAM test set, but on a combined dataset of ADAM, REFUGE and IDRID and that our multi-task model also predicts the optic disc center.

- <details>
    <summary>*details</summary>
    The ADAM dataset consists of 824 images sized 2124 x 2056 pixels and 376 images sized 1444 x 1444 pixels. The average side length of a squared image would be 0.5 * ((824(2124 + 2056)+376(1444*2)) / 1200) = 1887.59 pixels. Hence, the normalized distance of the winning model of the ADAM challenge is 18.5538 pixels / 1887.59 pixels = 0.0098.
    </details>

### Ref
This project uses substantial parts of a [tutorial](https://python.plainenglish.io/single-object-detection-with-pytorch-step-by-step-96430358ae9d) that has kindly given permission to use it. You can find the original code [here](https://github.com/dorzv/ComputerVision/blob/cc41b9d40af2b8b878f1352ec1308f031ad5b3f6/single_object_detection/Pytorch_Single_Object_Detection.ipynb).


### Cite