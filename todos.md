## TODOS

[ ] Add batch support:
    Allow both single image inputs as well as batches! If possible, also process in batches.

    [x] quality_prediction

    [x] fovea_od_localization

    [ ] registration

    [x] segmentation

    [x] circle_crop

[ ] Vessel Segmentation: Ask Patrick & Jerry if they could get rid of "bunch" dependency: "bunch" prevents the use of python >= 3.10. 31.1.205: PyPI does not accept bunch as a git dependency as there's an older version on PyPI. Hence now we clone it for the segmentation code on the fly.

[ ] Once Sarah has refactored her image cropping code to yield an image as output, add it to circle_crop, s.t. one can choose between the two algorithms

[ ] Once has DOI, add citation to single and main Readmes

[ ] Regularly check for TODO entries in any files
