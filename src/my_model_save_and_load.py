# USAGE
# python my_model.py

import json
import tensorflow as tf

# A Keras model consists of multiple components:
#    An architecture, or configuration, which specifies what layers the model contain, and how they're connected.
#    A set of weights values (the "state of the model").
#    An optimizer (defined by compiling the model).
#    A set of losses and metrics (defined by compiling the model or calling add_loss() or add_metric()).

def save_model_architecture(model_name, format_type):
    model = tf.keras.models.load_model(model_name)
    config = model.get_config()
    json_config = model.to_json()

    if format_type == "json":
        file_name = model_name + ".json"
    else:
        file_name = model_name + ".txt"

    with open(file_name, "w") as outfile:
        json.dump(json_config, outfile)

    with open(model_name + ".config.txt", "w") as text_file:
        text_file.write(str(config))


save_model_architecture("../models/model_CNN.h5", "json")
save_model_architecture("../models/model_NN.h5", "json")
