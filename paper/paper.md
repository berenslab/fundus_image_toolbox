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
    equal-contrib: false # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
  - name: Sarah Müller
    equal-contrib: false # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
  - name: Ziwei Huang
    equal-contrib: false # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
affiliations:
 - name: Hertie Institute for AI in Brain Health, University of Tübingen, Tübingen, Germany
   index: 1

date: 11 June 2024
bibliography: paper.bib
---

<!-- ![Fundus Image Toolbox Icon](../icon.svg) -->

# Summary
The Fundus Image Toolbox is a Python suite of tools for working with retinal fundus images. It includes quality prediction, localization for the fovea and optic disc centers, image registration, blood vessel segmentation, and fundus cropping functionalities. Additionally, it provides a collection of useful utilities for image manipulation and image based Pytorch models. The toolbox is designed to be flexible and easy to use. All tools can be installed as a whole or individually, depending on the user's needs. \autoref{fig:example} illustrates main functionalities.

# Statement of need
In the field of ophthalmological research, retinal fundus images frequently serve as resource for studying various eye diseases such as diabetic retinopathy, glaucoma, and age-related macular degeneration. Consequently, there is a great amount of research on tasks related to fundus image analysis. However, many works do not publish their source code and very few provide ready-to-use open-source tools for the community.

The Fundus Image Toolbox is developed to address this need within the research community. It offers a comprehensive set of tools for automated processing of retinal fundus images, covering a wide range of tasks: Fastly cropping fundus images to a circle, aligning fundus images, segmenting blood vessels, predicting the image quality, and localizing the fovea and optic disc centers. The quality prediction and localization models have been developed by the authors and allow for both prediction and retraining of the models. The other main functionalities are based on state-of-the-art methods from the literature [@fu2019; @liu2022; @huang2023; @koehler2024]. By providing an interface for these tasks, the toolbox aims to facilitate the development of new algorithms and models in the field of fundus image analysis. 

![Examples for main functionalities of the Fundus Image Toolbox. (a.) Fovea and optic disc localization. (b.) Vessel segmentation. (c.) Quality prediction. (d.) Registration.\label{fig:example}](fig.svg){ width=100% }

# Acknowledgements

This project was supported by the Hertie Foundation. J.G. received funding through the Else Kröner Medical Scientist Kolleg "ClinbrAIn: Artificial Intelligence for Clinical Brain Research”.

# References
