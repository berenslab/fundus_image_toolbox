VESSEL_SEGMENTATION_COMPONENT_NAME = "vessel_segmentation"
ENSEMBLE_WEIGHT_FILES = tuple(f"FRUNet_{i}.pth" for i in range(5))

# Same repository as the segmentation-quality-control git dependency in pyproject.toml.
VESSEL_WEIGHTS_GIT_URL = (
    "https://github.com/juliusge/MIDL24-segmentation_quality_control.git"
)
# Pin a commit so weight downloads stay reproducible; bump when updating the dependency.
VESSEL_WEIGHTS_GIT_REF = "eb423be5e11198ac9729282f3cab54022638c9fb"
VESSEL_WEIGHTS_GIT_SUBDIR = "trained"
