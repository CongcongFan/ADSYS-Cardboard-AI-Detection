import ollama

def set_ollama_parameter(model_name, parameter, value):
    """Set Ollama model parameter via API using options"""
    client = ollama.Client()
    
    try:
        # Create options dictionary with the parameter
        options = {parameter: int(value) if value.isdigit() else value}
        
        # Test with a simple prompt to verify the parameter is applied
        response = client.generate(
            model=model_name,
            prompt="test",
            stream=False,
            options=options
        )
        print(f"Success: Set {parameter}={value} for {model_name}")
        print(f"Options applied: {options}")
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def get_ollama_models():
    """List available models"""
    client = ollama.Client()
    try:
        models_response = client.list()
        models = models_response.get('models', [])
        print("Available models:")
        model_names = []
        for model in models:
            # Extract model name from the model object
            if hasattr(model, 'model'):
                name = model.model
            elif isinstance(model, dict) and 'name' in model:
                name = model['name']
            else:
                name = str(model)
            print(f"  - {name}")
            model_names.append(name)
        return model_names
    except Exception as e:
        print(f"Error getting models: {str(e)}")
        return []

if __name__ == "__main__":
    # Example usage
    print("Ollama Parameter Setter")
    
    # List available models
    models = get_ollama_models()
    
    if models:
        # Set num_gpu parameter
        model_name = "qwen2.5vl:3b"  # Change this to your model
        if model_name in models:
            set_ollama_parameter(model_name, "num_gpu", "999")
        else:
            print(f"Warning: Model {model_name} not found. Available models: {models}")
    else:
        print("No models available or Ollama not running")