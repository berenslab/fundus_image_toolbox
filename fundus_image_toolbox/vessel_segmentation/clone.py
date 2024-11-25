import os


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
    # Clone Segmentation Quality Control repo
    os.makedirs(target_dir, exist_ok=True)
    if not os.listdir(target_dir):
        print(
            f"The vessel segmentation folder is missing. Downloading it from {link}..."
        )
        pwd = os.getcwd()
        os.system(f"git clone {link} {target_dir}")
        os.system(
            f"cd {target_dir} && git checkout {branch} && git reset --hard {commit}"
        )
        try:
            remove_folder(os.path.join(target_dir, ".git"))
        except:
            pass
        os.chdir(pwd)

        print("Adjusting imports...")
        add_dots_to_imports_in_folder(target_dir)
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
                    print(os.path.join(this_dir, fname + ".py"))
                    is_local_file = True
                # Check if is a local folder in parent directory
                elif os.path.exists(os.path.join(par_dir, fname)):
                    print(os.path.join(par_dir, fname))
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

                print("parents: ", parents)
                print("imports: ", imports)

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
                print("########\n", file)
                add_dots_to_imports(os.path.join(root, file))
