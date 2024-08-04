import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
from functools import partial
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from rawdata_divide import matrices_by_group
from rawdata_divide import columns_to_select_by_group
import numpy as np
import pandas as pd
from os.path import join
import subprocess
def run_script(script_name):
    try:
        subprocess.run(['python', script_name], check=True)
        print(f"{script_name} executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running {script_name}: {e}")
if __name__ == "__main__":
    scripts = [
        'identification.py',
        'rawdata_divide.py',
        'Runmodel.py',
        'divide_reconstrtor.py',
        'whole_reconstrtor.py'
    ]
    for script in scripts:
        run_script(script)
