import os
from pathlib import Path
import subprocess


def remove_folder(target_dir):
    for root, dirs, files in os.walk(target_dir, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(target_dir)


def clone_repo(
    link="https://github.com/berenslab/MIDL24-segmentation_quality_control",
    commit="995a173c1c1735b620c14b2268ce3ca408c10ca1",
    branch="main",
    target_dir=f"./segmentation",
):
    target_path = Path(target_dir).resolve()
    target_path.mkdir(parents=True, exist_ok=True)

    if not any(target_path.iterdir()):
        print(f"Module missing, downloading it from {link}...")
        
        # Clone the repository
        subprocess.run(
            ["git", "clone", link, str(target_path)],
            check=True,
            capture_output=True,
            text=True
        )
        
        # Checkout specific branch and commit
        subprocess.run(
            ["git", "checkout", branch],
            cwd=str(target_path),
            check=True,
            capture_output=True,
            text=True
        )
        
        subprocess.run(
            ["git", "reset", "--hard", commit],
            cwd=str(target_path),
            check=True,
            capture_output=True,
            text=True
        )
        
        # Remove .git folder
        git_folder = target_path / ".git"
        if git_folder.exists():
            try:
                remove_folder(str(git_folder))
            except Exception:
                pass

def adjust_imports(target_dir):
    # Make imports of segmentation module relative
    if os.path.exists(target_dir):
        print("Adjusting imports...")
        add_dots_to_imports_in_folder(target_dir)
        replace_args(target_dir)
        print("Done.")

# Make imports relative
def add_dots_to_imports(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
    with open(file_path, "w") as file:
        for line in lines:
            is_from_import = (
                line.split()[0] == "from" if len(line.split()) > 1 else False
            )
            is_import = line.split()[0] == "import" if len(line.split()) > 1 else False
            is_local_file = False
            is_local_folder = False

            if is_from_import or is_import:
                this_dir = os.path.dirname(file_path)
                par_dir = os.path.dirname(this_dir)
                fname = line.split()[1]
                if fname.startswith("."):
                    file.write(line)
                    continue
                fname = fname.split(".")[0]
                # print(fname)

                # Check if is a local file in same directory
                if os.path.exists(os.path.join(this_dir, fname)):
                    # print(os.path.join(this_dir, fname + ".py"))
                    is_local_file = True
                # Check if is a local folder in parent directory
                elif os.path.exists(os.path.join(par_dir, fname)):
                    # print(os.path.join(par_dir, fname))
                    is_local_folder = True

            c = ".." if is_local_folder else "."
            if is_local_file or is_local_folder:
                terms = (
                    line.replace("import", ".")
                    .replace("from", "")
                    .replace("\n", "")
                    .split(".")
                )
                terms = [t.replace(" ", "") for t in terms]
                parents = terms[:-1] if len(terms) > 1 else []
                imports = terms[-1].split(",")
                imports = [i.replace(" ", "") for i in imports]

                # print("parents: ", parents)
                # print("imports: ", imports)

                if len(parents) > 0:
                    file.write(
                        f"from {c}{'.'.join(parents)} import {', '.join(imports)}\n"
                    )
                else:
                    file.write(f"from {c} import {', '.join(imports)}\n")

            else:
                file.write(line)


def add_dots_to_imports_in_folder(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".py"):
                # print("########\n", file)
                add_dots_to_imports(os.path.join(root, file))

def replace_args(folder_path):
    # Prevents deprecation error from timm module and torch error when loading model from bunch object
    args = [
        ("timm.models.layers", "timm.layers"),
        ("checkpoint= torch.load(predictor_path, map_location=device)", "checkpoint= torch.load(predictor_path, map_location=device, weights_only=False)")
    ]
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".py"):
                with open(os.path.join(root, file), "r") as f:
                    lines = f.readlines()
                with open(os.path.join(root, file), "w") as f:
                    for line in lines:
                        for arg in args:
                            line = line.replace(arg[0], arg[1])
                        f.write(line)
