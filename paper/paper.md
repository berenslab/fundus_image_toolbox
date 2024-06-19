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

date: 20 June 2024
bibliography: paper.bib
---

<!-- ![Fundus Image Toolbox Icon](../icon.svg) -->

# Summary
The Fundus Image Toolbox is a Python suite of tools for working with retinal fundus images. It includes quality prediction, localization for the fovea and optic disc centers, image registration, blood vessel segmentation, and fundus cropping functionalities. Additionally, it provides a collection of useful utilities for image manipulation and image based PyTorch models. The toolbox is designed to be flexible and easy to use. All tools can be installed as a whole or individually, depending on the user's needs. \autoref{fig:example} illustrates main functionalities. Find the toolbox here: [github.com/berenslab/fundus_image_toolbox](https://github.com/berenslab/fundus_image_toolbox)

# Statement of need
In ophthalmic research, retinal fundus images are often used as a resource for studying various eye diseases such as diabetic retinopathy, glaucoma and age-related macular degeneration. Consequently, there is a large amount of research on machine learning for fundus image analysis. However, many of the works do not publish their source code, and very few of them provide ready-to-use open source tools to the community.

The Fundus Image Toolbox has been developed to address this need within the medical image analysis community. It offers a comprehensive set of tools for automated processing of retinal fundus images, covering a wide range of tasks: 

- Quality prediction. We trained an ensemble of ResNets and EfficientNets on the combined DeepDRiD and DrimDB datasets to predict the gradeability of fundus images. Both datasets are publicly available. The model ensemble achieved an accuracy and ROCAUC of 0.78 and 0.84 on a DeepDRiD test split and 1.0 and 1.0 on a DrimDB test split.
- Fovea and optic disc localization. Prediction of fovea and optic disc center coordinates using a multi-task EfficientNet. We trained the model on the combined ADAM, REFUGE and IDRID datasets, which are publicly available. On our test split, the model achieved a mean distance to the fovea and optic disc targets of 0.0088 (normalized to a value in [0,1]). This corresponds to a distance of 3,08 pixels in the 350 x 350 pixel images used for training and testing.
- Registration. Alignment of a fundus photograph to another fundus photograph of the same eye using SuperRetina: a keypoint-based deep learning model that produced registrations of at least acceptable quality in 98.5 % of the cases on the test split of the FIRE dataset [@liu2022].
- Vessel segmentation. Segmentation of blood vessels in a fundus image using an ensemble of FR-U-Nets [@koehler2024]. The ensemble achieved an average Dice score of 0.887 on the test split of the FIVES dataset.
- Circle crop. Fastly center fundus images and crop to a circle [@fu2019].
- Utilities. A collection of useful other tools for fundus image processing.

The methods all accept paths to images, standard image types and batches thereof and where possible, image batches are efficiently processed as such. This allows the tools to be seamlessly combined into a processing pipeline. The quality prediction and localization models allow for both prediction and retraining of the models while the other main functionalities based on state-of-the-art methods from the literature provide prediction methods. By providing an interface for these tasks, the toolbox aims to facilitate the development of new algorithms and models in the field of fundus image analysis. AutoMorph is the closest related work [@zhou2022], which provides a distinct and smaller set of tools for fundus image processing.

![Examples for main functionalities of the Fundus Image Toolbox. (a.) Fovea and optic disc localization. (b.) Vessel segmentation. (c.) Quality prediction. (d.) Registration.\label{fig:example}](fig.svg){ width=100% }

# Acknowledgements
We thank Ziwei Huang for reviewing the package. This project was supported by the Hertie Foundation. JG received funding through the Else Kröner Medical Scientist Kolleg "ClinbrAIn: Artificial Intelligence for Clinical Brain Research”. The authors thank the International Max Planck Research School for Intelligent Systems (IMPRS-IS) for supporting SM.

# References
