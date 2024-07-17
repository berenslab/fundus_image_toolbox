---
title: 'Fundus Image Toolbox: A Python package for fundus image processing'
tags:
  - Python
  - fundus image
  - retina
  - registration
  - optic disc
  - fovea
  - vessel segmentation
  - quality prediction
  - circle crop
authors:
  - name: Julius Gervelmeyer
    orcid: 0009-0007-7286-0017
    equal-contrib: false
    affiliation: 1
  - name: Sarah Müller
    orcid: 0000-0003-1500-8673
    equal-contrib: false
    affiliation: 1
  - name: Philipp Berens
    orcid: 0000-0002-0199-4727
    equal-contrib: false
    affiliation: 1
affiliations:
 - name: Hertie Institute for AI in Brain Health, University of Tübingen, Tübingen, Germany
   index: 1

date: 20 July 2024
bibliography: paper.bib
---

<!-- ![Fundus Image Toolbox Icon](../icon.svg) -->

# Summary
The Fundus Image Toolbox is an open source Python suite of tools for working with retinal fundus images. It includes quality prediction, fovea and optic disc center localization, blood vessel segmentation, image registration, and fundus cropping functions. It also provides a collection of useful utilities for image manipulation and image-based PyTorch models. The toolbox is designed to be flexible and easy to use, thus helping to speed up research workflows. All tools can be installed as a whole or individually, depending on the user's needs. \autoref{fig:example} illustrates the main functionalities. 
Find the toolbox at [https://github.com/berenslab/fundus_image_toolbox](https://github.com/berenslab/fundus_image_toolbox).

# Statement of need
In ophthalmic research, retinal fundus images are often used as a resource for studying various eye diseases such as diabetic retinopathy, glaucoma and age-related macular degeneration. Consequently, there is a large amount of research on machine learning for fundus image analysis. However, many of the works do not publish their source code, and very few of them provide ready-to-use open source preprocessing tools to the community.

The Fundus Image Toolbox was developed to address this need within the medical image analysis community. It offers a comprehensive set of tools for automated processing of retinal fundus images, covering a wide range of tasks (see Tools). The methods accept paths to images, standard image types (e.g., images loaded with Pillow, OpenCV, or Matplotlib) and batches thereof and where possible, image batches are efficiently processed as such. This allows the tools to be seamlessly combined into a processing pipeline. The quality prediction and localization models have been developed by the authors and allow for both prediction and retraining, while the other main functionalities are based on state-of-the-art methods from the literature that are applicable for inference. By providing an interface for these tasks, the toolbox facilitates the development of new algorithms and models in the field of fundus image analysis. AutoMorph is the closest related work [@zhou2022], which provides a distinct and smaller set of tools for fundus image processing.

# Tools
The main functionalities of the Fundus Image Toolbox are:

- Quality prediction (\autoref{fig:example}a). We trained an ensemble of ResNets and EfficientNets on the combined DeepDRiD and DrimDB datasets [@deepdrid;@drimdb] to predict the gradeability of fundus images. The datasets are publicly available and comprise images of retinas with diabetic retinopathy, healthy retinas and outliers such as outer eye images. The model ensemble achieved an accuracy of 0.78 and an area under the receiver operating characteristic curve (AUROC) of 0.84 on a DeepDRiD test split, surpassing the previous best model evaluated on DeepDRiD [@tummala2023]. Further, on a DrimDB test split, our model acchieved an accuracy and AUROC of 1.0 and 1.0, respectively.
- Fovea and optic disc localization (\autoref{fig:example}b). The center coordinates of the fovea and optic disc can be predicted using a multitask EfficientNet model. We trained the model on the combined ADAM, REFUGE and IDRID datasets which include images from eyes with age-related macular degeneration, glaucoma, diabetic retinopathy and healthy retinas [@adam;@refuge;@idrid]. All datasets are publicly available. On our test split, the model achieved an average distance to the fovea and optic disc targets of 0.88 % of the image size. This corresponds to a mean distance of 3.08 pixels in the 350 x 350 pixel images used for training and testing.
- Vessel segmentation (\autoref{fig:example}c). The segmentation method produces a mask of blood vessels in a fundus image using an ensemble of FR-UNets. The ensemble achieved an average Dice score of 0.887 on the test split of the FIVES dataset [@koehler2024]. FIVES includes images with age-related macular degeneration, glaucoma, diabetic retinopathy and healthy retinas [@fives].
- Registration (\autoref{fig:example}d). Two fundus images of the same eye can be aligned using SuperRetina [@liu2022]. The deep learning based model detects key points on the vessel trees of the two images and matches them. This results in a registered version of the second image that is aligned with the first. SuperRetina produced registrations of at least acceptable quality in 98.5 % of the cases on the test split of the FIRE dataset [@fire].
- Circle cropping. The OpenCV-based implementation fastly crops the circular background from a fundus image. The circle is further resized to touch the edges of the image, which centers the fundus [@fu2019].


![Examples for main functionalities of the Fundus Image Toolbox. (a.) Fovea and optic disc localization. (b.) Quality prediction. (c.) Vessel segmentation. (d.) Registration.\label{fig:example}](fig3.pdf){ width=100% }

# Acknowledgements
We thank Ziwei Huang for reviewing the package and Murat Seçkin Ayhan for inspiring the development of the quality prediction model. This project was supported by the Hertie Foundation. JG received funding through the Else Kröner Medical Scientist Kolleg "ClinbrAIn: Artificial Intelligence for Clinical Brain Research”. The authors thank the International Max Planck Research School for Intelligent Systems (IMPRS-IS) for supporting SM.

# References
