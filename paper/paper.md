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
The Fundus Image Toolbox is a Python suite of tools for working with retinal fundus images. It includes quality prediction, localization for the fovea and optic disc centers, image registration, blood vessel segmentation, and fundus cropping functionalities. Additionally, it provides a collection of useful utilities for image manipulation and image based Pytorch models. The toolbox is designed to be flexible and easy to use. All tools can be installed as a whole or individually, depending on the user's needs. \autoref{fig:example} illustrates main functionalities. Find the toolbox here: [github.com/berenslab/fundus_image_toolbox](https://github.com/berenslab/fundus_image_toolbox)

# Statement of need
In ophthalmic research, retinal fundus images are often used as a resource for studying various eye diseases such as diabetic retinopathy, glaucoma and age-related macular degeneration. Consequently, there is a large amount of research on machine learning for fundus image analysis. However, many of the works do not publish their source code, and very few of them provide ready-to-use open source tools to the community.

The Fundus Image Toolbox has been developed to address this need within the medical image analysis community. It offers a comprehensive set of tools for automated processing of retinal fundus images, covering a wide range of tasks: Fastly cropping fundus images to a circle, aligning fundus images, segmenting blood vessels, predicting the image quality, and localizing the fovea and optic disc centers. The methods accept all standard image types and batches thereof and whenever possible, image batches are efficiently processed as such. This allows the tools to be seamlessly combined into a processing pipeline. The quality prediction and localization models have been developed by the authors and allow for both prediction and retraining of the models. The other main functionalities are based on state-of-the-art methods from the literature [@fu2019; @liu2022; @koehler2024]. By providing an interface for these tasks, the toolbox aims to facilitate the development of new algorithms and models in the field of fundus image analysis. AutoMorph is the closest related work [@zhou2022], which provides a distinct and smaller set of tools for fundus image processing.

![Examples for main functionalities of the Fundus Image Toolbox. (a.) Fovea and optic disc localization. (b.) Vessel segmentation. (c.) Quality prediction. (d.) Registration.\label{fig:example}](fig.svg){ width=100% }

# Acknowledgements
We thank Ziwei Huang for reviewing the package. This project was supported by the Hertie Foundation. JG received funding through the Else Kröner Medical Scientist Kolleg "ClinbrAIn: Artificial Intelligence for Clinical Brain Research”. The authors thank the International Max Planck Research School for Intelligent Systems (IMPRS-IS) for supporting SM.

# References
