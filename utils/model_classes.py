from tensorflow.keras.models import load_model

def get_model_num_classes(model_path):
    
    """
    Load a Keras model and retrieve the number of output classes.

    Args:
        model_path (str): Path to the saved Keras model file.

    Returns:
        int: The number of classes in the model's output layer.
    """

    try:
        model = load_model(model_path)

        output_layer = model.layers[-1]  
        num_classes = output_layer.units   

        return num_classes

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

model_path = '/path/to/model.keras'

num_classes = get_model_num_classes(model_path)
if num_classes is not None:
    print(f"Number of classes in the model: {num_classes}")
