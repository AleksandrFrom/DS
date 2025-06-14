import pickle
import numpy as np

with open('autoclave_model_rf_density.pkl', 'rb') as file:
    density_model = pickle.load(file)

with open('autoclave_model_rf_thikness.pkl', 'rb') as file:
    thikness_model = pickle.load(file)

with open('autoclave_model_rf_strength.pkl', 'rb') as file:
    strength_model = pickle.load(file)

with open('autoclave_model_rf_module.pkl', 'rb') as file:
    module_model = pickle.load(file)

with open('autoclave_model_rf_lss.pkl', 'rb') as file:
    lss_model = pickle.load(file)

with open('autoclave_model_rf_Tg.pkl', 'rb') as file:
    tg_model = pickle.load(file)

#x = np.array([[190, 1.779, 4.6, 265, 1.7, 1.3, 20, 210, 333, 37.40, 32.33, 12.8, 149.0]])

