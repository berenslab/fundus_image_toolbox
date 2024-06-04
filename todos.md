## TODOS

[x] circle_crop: Add Example image to Readme

[x] fovea_od_localization: Add Example image to Readme

[x] quality_prediction: Add Example image to Readme

[x] registration: Add Example image to Readme

[x] segmentation: Add Example image to Readme

[x] utilities: Add Readme, describe each tool

[x] Add / fetch example fundus image(s) for the usage.ipynb files

[ ] State all external paper, data and fundus package dependencies, including the License stuff

    [(x)] external datasets (none included here, but have to be downloaded for optional training): DeepDRiD, DrimDB, ADAM, REFUGE, IDRID

    [ ] external code: SuperRetina (done), VesselSegmentation, MultiLevelSplit, Transforms from PytorchClassification, ImbalancedDatasetSampler

[x] Include an example image in this main readme for every package

[ ] Batch support (accept batches and if possible, also process in batches):

    [x] quality_prediction: Allow both single image inputs as well as batches!

    [x] fovea_od_localization: Allow both single image inputs as well as batches!

    [ ] registration: Allow both single image inputs as well as batches!

    [x] segmentation: Allow both single image inputs as well as batches!

    [?] circle_crop: Allow both single image inputs as well as batches!

    [?] utilities: Allow both single image inputs as well as batches!

[x] Test if quality model trains after changing model.predict_from_batch function

[x] segmentation: Test if fresh install clones and fixes the code

[x] Add overview figure to top of readme, or symbol or so

[x] Unify load_model, load_ensemble, etc functions -> load_registration_model, load_segmentation_ensemble, etc
    [x] in general usage.ipynb 

    [x] in general readme

    [x] tested

[ ] vessel: Ask Patrick & Jerry if they could get rid of "bunch" dependency: "bunch" prevents the use of python >= 3.10

[ ] vessel: Ask Patrick & Jerry if I can include their code directly. That would need huge refactoring though as the imports are not package suited and changing them leads to circular import errors :S

[ ] Once Sarah has refactored her image cropping code to yield an image as output, add it to circle_crop, s.t. one can choose between the two algorithms

[ ] Once has DOI, add citation to single and main Readmes

[ ] write small paper.md for JOSS

[ ] write more unit tests