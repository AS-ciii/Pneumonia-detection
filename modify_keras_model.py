import h5py

# Path to your Keras model file
file_path = "./models/CT_classification_model.h5"

# Open and modify the file
f = h5py.File(file_path, mode="r+")
model_config_string = f.attrs.get("model_config")

# Check and modify if necessary
if model_config_string.find('"groups": 1,') != -1:
    model_config_string = model_config_string.replace('"groups": 1,', '')
    f.attrs.modify('model_config', model_config_string)
    f.flush()
    model_config_string = f.attrs.get("model_config")
    assert model_config_string.find('"groups": 1,') == -1

f.close()
print("Model configuration updated successfully.")
