## TODOS

[ ] Add batch support:
    Allow both single image inputs as well as batches! If possible, also process in batches.

    [x] quality_prediction

    [x] fovea_od_localization

    [ ] registration

    [x] segmentation

    [x] circle_crop

[ ] Vessel Segmentation: Ask Patrick Köhler & Jeremiah Fadugba if they could get rid of the "bunch" dependency: "bunch" prevents the use of python >= 3.10. The model weights are stored in a bunch object. 31.1.2025: PyPI does not accept bunch as a git dependency as there's an older version on PyPI. Hence now we clone it for the segmentation code on the fly.

[ ] Once Sarah has refactored her image cropping code to yield an image as output, add it to circle_crop, s.t. one can choose between the two algorithms

[ ] Regularly check for TODO entries in any files

[ ] Circle cropping. Ifeoma Nwabufo found out that it can be improved by adding a Gaussion filter to the input before fitting the circle. 

[ ] Circle cropping. Ifeoma Nwabufo reported that some landscape images cannot be circle cropped -- only if square cropping them first. This is odd, should look into it!
