import os
from setuptools import setup, find_packages

project = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
# about = __import__(project).about

with open("Readme.md", "r", encoding='utf8') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name=project,
    version="0.0.1",
    author="Julius Gervelmeyer et al.",
    author_email="Julius.Gervelmeyer@uni-tuebingen.de",
    description=long_description.split('\n')[0],
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url=None,
    packages=find_packages(),
    install_requires=required,
    package_dir={'': '.'},
    python_requires='>=3.9.0, <3.10',
)

def clone_repo():
    # Clone Segmentation Quality Control repo
    link = "https://github.com/berenslab/MIDL24-segmentation_quality_control"
    commit = "995a173c1c1735b620c14b2268ce3ca408c10ca1"
    branch = "main"
    target_dir = f"./{project}/segmentation"
    os.makedirs(target_dir, exist_ok=True)
    if not os.listdir(target_dir):
        pwd = os.getcwd()
        os.system(f"git clone {link} {target_dir}")
        os.system(f"cd {target_dir} && git checkout {branch} && git reset --hard {commit}")
        os.system(f"rm -rf {target_dir}/.git")
        os.chdir(pwd)
        print("Repo cloned successfully.")

        print("Adjusting imports...")
        add_dots_to_imports_in_folder(target_dir)
        print("Done.")

# Make imports relative
def add_dots_to_imports(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    with open(file_path, 'w') as file:
        for line in lines:
            is_from_import = line.split()[0] == 'from' if len(line.split()) > 1 else False
            is_import = line.split()[0] == 'import' if len(line.split()) > 1 else False
            is_local_file = False
            is_local_folder = False
            
            if is_from_import or is_import:
                this_dir = os.path.dirname(file_path)
                par_dir = os.path.dirname(this_dir)
                fname = line.split()[1]
                if fname.startswith('.'):
                    file.write(line)
                    continue
                fname = fname.split('.')[0]
                # print(fname)

                # Check if is a local file in same directory
                if os.path.exists(os.path.join(this_dir, fname)):
                    print(os.path.join(this_dir, fname + '.py'))
                    is_local_file = True
                # Check if is a local folder in parent directory
                elif os.path.exists(os.path.join(par_dir, fname)):
                    print(os.path.join(par_dir, fname))
                    is_local_folder = True

            c = ".." if is_local_folder else "."
            if is_local_file or is_local_folder:
                terms = line.replace("import", ".").replace("from", "").replace("\n", "").split(".")
                terms = [t.replace(" ", "") for t in terms]
                parents = terms[:-1] if len(terms) > 1 else []
                imports = terms[-1].split(",")
                imports = [i.replace(" ", "") for i in imports]

                print("parents: ", parents)
                print("imports: ", imports)
                
                if len(parents) > 0:
                    file.write(f"from {c}{'.'.join(parents)} import {', '.join(imports)}\n")
                else:
                    file.write(f"from {c} import {', '.join(imports)}\n")
                
            else:
                file.write(line)

def add_dots_to_imports_in_folder(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.py'):
                print("########\n", file)
                add_dots_to_imports(os.path.join(root, file))


# Clone the repo
clone_repo()
add_dots_to_imports_in_folder(f"./{project}/segmentation")