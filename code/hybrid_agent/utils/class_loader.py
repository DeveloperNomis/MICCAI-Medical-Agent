import yaml

def load_class_names_from_yaml(yaml_path: str):

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    names_dict = data.get("names", {})

    # Sort after index
    class_names = [
        names_dict[i]
        for i in sorted(names_dict.keys())
    ]

    return class_names