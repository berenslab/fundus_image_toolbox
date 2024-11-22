import os
from setuptools import setup, find_packages

root = os.path.dirname(os.path.realpath(__file__))
project = os.path.basename(root)
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

## The current setup includes the SuperRetina repo already (May 2024), as the authors allowed
##  any usage of their code. If this changed or you want the latest commit, uncomment this code to
##  - clone the latest commit of the SuperRetina repo
##  - adjust all imports for package usage
##  - download weights from their google drive (as opposed to re-upload on zenodo)
##  and re-install the package with `pip install -e .` in the root directory of this registration project.
#
# # Clone SuperRetina repo
# link = "https://github.com/ruc-aimc-lab/SuperRetina"
# commit = "338f041cc2ce86f39623e7da950b14f33bbc25df"
# branch = "main"
# target_dir = f"./{project}/SuperRetina/"
# if not os.path.exists(target_dir) or not os.listdir(target_dir):
#     pwd = os.getcwd()
#     os.system(f"git clone {link} {target_dir}")
#     os.system(f"cd {target_dir} && git checkout {branch} && git reset --hard {commit}")
#     os.system(f"rm -rf {target_dir}/.git")
#     os.chdir(pwd)
#     print("Repo cloned successfully.")

# # Make imports relative
# def add_dots_to_imports(file_path):
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
#     with open(file_path, 'w') as file:
#         for line in lines:
#             is_from_import = line.split()[0] == 'from' if len(line.split()) > 1 else False
#             is_import = line.split()[0] == 'import' if len(line.split()) > 1 else False
#             is_local_file = False
#             is_local_folder = False
            
#             if is_from_import or is_import:
#                 this_dir = os.path.dirname(file_path)
#                 par_dir = os.path.dirname(this_dir)
#                 fname = line.split()[1]
#                 if fname.startswith('.'):
#                     file.write(line)
#                     continue
#                 fname = fname.split('.')[0]
#                 # print(fname)

#                 # Check if is a local file in same directory
#                 if os.path.exists(os.path.join(this_dir, fname)):
#                     print(os.path.join(this_dir, fname + '.py'))
#                     is_local_file = True
#                 # Check if is a local folder in parent directory
#                 elif os.path.exists(os.path.join(par_dir, fname)):
#                     print(os.path.join(par_dir, fname))
#                     is_local_folder = True

#             c = ".." if is_local_folder else "."
#             if is_local_file or is_local_folder:
#                 terms = line.replace("import", ".").replace("from", "").replace("\n", "").split(".")
#                 terms = [t.replace(" ", "") for t in terms]
#                 parents = terms[:-1] if len(terms) > 1 else []
#                 imports = terms[-1].split(",")
#                 imports = [i.replace(" ", "") for i in imports]

#                 print("parents: ", parents)
#                 print("imports: ", imports)
                
#                 if len(parents) > 0:
#                     file.write(f"from {c}{'.'.join(parents)} import {', '.join(imports)}\n")
#                 else:
#                     file.write(f"from {c} import {', '.join(imports)}\n")
                
#             else:
#                 file.write(line)

# def add_dots_to_imports_in_folder(folder_path):
#     for root, _, files in os.walk(folder_path):
#         for file in files:
#             if file.endswith('.py'):
#                 print("########\n", file)
#                 add_dots_to_imports(os.path.join(root, file))

# print("Adjusting imports...")
# add_dots_to_imports_in_folder(f'./{project}/SuperRetina')
# print("Done.")

# # Download weights
# #   original drive link from repo: https://drive.google.com/drive/folders/1h-MH3wEiN7BoLyMRjF1OAwABKqq6gVFL
# #   retrieved cli link using $$("[data-id]").map((el) => 'https://drive.google.com/uc?id=' + el.getAttribute('data-id')).join(" ")
# target_dir = f"./{project}/SuperRetina/save"
# if not os.path.exists(os.path.join(target_dir,"SuperRetina.pth")):
#     weights = "https://drive.google.com/uc?id=1OL7LhLLRSBW72AHOyAwqjCeKV-XpP2GW"
#     e = os.system(f"gdown {weights} -O {target_dir}")
#     # Use if fetching a folder:
#     # os.system(f"mv {target_dir}/SuperRetina/SuperRetina.pth {target_dir}/SuperRetina.pth")
#     # os.system(f"rm -r {target_dir}/SuperRetina")
#     if e == 0:
#         print("Weights downloaded successfully.")
#     else:
#         print("Weights could not be downloaded. This happens if Google blocked automatic access. Please download the weights manually from the link below and place it in the folder 'registration/SuperRetina/save' with the name 'SuperRetina.pth'.")
#         print("Link: https://drive.google.com/uc?id=1OL7LhLLRSBW72AHOyAwqjCeKV-XpP2GW")