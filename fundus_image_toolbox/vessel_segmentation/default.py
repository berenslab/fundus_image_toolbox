VESSEL_SEGMENTATION_COMPONENT_NAME = "vessel_segmentation"
ENSEMBLE_WEIGHT_FILES = tuple(f"FRUNet_{i}.pth" for i in range(5))

# Same repository as the segmentation-quality-control git dependency in pyproject.toml.
VESSEL_WEIGHTS_GIT_URL = (
    "https://github.com/berenslab/MIDL24-segmentation_quality_control.git"
)
# Pin a commit so weight downloads stay reproducible; bump when updating the dependency.
VESSEL_WEIGHTS_GIT_REF = "6ec927161c4db9f727d6213395227c6beaf778af"
VESSEL_WEIGHTS_GIT_SUBDIR = "trained"
