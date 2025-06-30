from pathlib import Path
from src.utils.main_utils import read_yaml


params = read_yaml(Path("config/params.yaml"))
a = params.params.CLASSES
print(a)