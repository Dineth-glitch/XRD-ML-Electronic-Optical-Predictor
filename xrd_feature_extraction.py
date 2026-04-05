import numpy as np
import pandas as pd
from tqdm import tqdm
from mp_api.client import MPRester
from pymatgen.analysis.diffraction.xrd import XRDCalculator

CSV_FILE = "ml_materials_10000.csv"
API_KEY = "YOUR_MATERIALS_PROJECT_API_KEY"

NUM_POINTS = 8192
TWO_THETA_RANGE = (5, 90)

df = pd.read_csv(CSV_FILE)
material_ids = df["material_id"].tolist()

print("Number of materials:", len(material_ids))

mpr = MPRester(API_KEY)
xrd_calc = XRDCalculator()

def get_fixed_length_xrd(structure, num_points=8192, two_theta_range=(5, 90)):
    pattern = xrd_calc.get_pattern(structure, two_theta_range=two_theta_range)

    two_theta = np.array(pattern.x)
    intensity = np.array(pattern.y)

    grid = np.linspace(two_theta_range[0], two_theta_range[1], num_points)
    fixed_intensity = np.interp(grid, two_theta, intensity)

    fixed_intensity /= (fixed_intensity.max() + 1e-8)
    return fixed_intensity

xrd_features = []
failed_ids = []

for mpid in tqdm(material_ids, desc="Extracting XRD"):
    try:
        structure = mpr.get_structure_by_material_id(mpid)
        xrd_vec = get_fixed_length_xrd(structure, NUM_POINTS, TWO_THETA_RANGE)
        xrd_features.append(xrd_vec)
    except Exception as e:
        print(f"Failed for {mpid}: {e}")
        failed_ids.append(mpid)
        xrd_features.append(np.zeros(NUM_POINTS))

xrd_array = np.array(xrd_features)
np.save("xrd_features.npy", xrd_array)

print("XRD extraction completed.")
print("Shape:", xrd_array.shape)
print("Failed IDs:", failed_ids)
