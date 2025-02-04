from flask import Flask, request, jsonify # Framework Flask, request gets the JSON data from the API, jsonify converts the python back to json format
from flask_restful import Api, Resource 
from flask_swagger_ui import get_swaggerui_blueprint # Used for easy API testing with Swagger
from transformers import AutoModelForCausalLM, AutoTokenizer # Loading the GBT-2 model
import torch
from werkzeug.exceptions import HTTPException


# Initialize Flask app, RESTful API 
app = Flask(__name__)
api = Api(app)

# Load GPT-2 model and tokenizer
# A tokenizer converts text into numerical tokens that the model understands
model_identifier = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_identifier)
model = AutoModelForCausalLM.from_pretrained(model_identifier)

# Set a pad token, as end-of-sequence token (GPT-2 doesn't have one by default)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Swagger UI setup
# Can be found at the website: http://127.0.0.1:5000/swagger/#/
SWAGGER_URL = "/swagger"
API_URL = "/static/swagger.json"
swagger_ui_blueprint = get_swaggerui_blueprint(SWAGGER_URL, API_URL)
app.register_blueprint(swagger_ui_blueprint, url_prefix=SWAGGER_URL)

# Predict endpoint as a POST request in this Flask-RESTful API
class Predict(Resource):
    def post(self):
        try:
            data = request.get_json()

            # Error handling
            if not data: # Check if the request body contains data
                raise ValueError("No input data provided.")
            if "text" not in data: # Check for the 'text' field in the request
                raise ValueError("Missing 'text' field in the input.")

            prompt = data["text"]
            
            if not isinstance(prompt, str): # Check if 'text' is a string
                raise ValueError("The 'text' field must be a string.")
            if len(prompt.strip()) == 0: # Check if the 'text' is empty
                raise ValueError("The 'text' field cannot be empty.")

            max_length = data.get("max_length", 100)

            if not isinstance(max_length, int): # Ensure max_length is an integer
                raise ValueError("The 'max_length' field must be an integer.")
            if max_length <= 0 or max_length > 500: # Check if max-length is within a reasonable range (could be up to 1024 which is GBT-2 usual max_length)
                raise ValueError("The 'max_length' has to be between 0-500 tokens, this value is not reasonable.")

            # Tokenize and attention mask
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask  # Ensure padding token is ignored

            # Using the GBT-2 model to generate the text
            with torch.no_grad(): # speed up inference by disable gradient computation
                output_ids = model.generate(
                    input_ids, 
                    attention_mask=attention_mask,  # Prevents unwanted padding - focus on real text only
                    max_length=max_length,
                    do_sample=True,  # Enable sampling for diverse results
                    top_p=0.9,  # Controls nucleus sampling (remove the low-probability words)
                    temperature=0.7,  # Controls randomness (higher = more random)
                    pad_token_id=tokenizer.eos_token_id  # Avoid padding issues
                )

            # Converts the token back to text
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True) 
            return {"generated_text": generated_text}, 200

        except ValueError as e: # Handle ValueErrors for bad inputs
            return {"error": str(e)}, 400
        
        except RuntimeError as e: # Handle torch-related errors (model inference issues)
            return {"error": "Model inference failed. Please try again later."}, 500

        except Exception as e: # Handle all other exceptions
            return {"error": "An unexpected error occurred. Please try again later."}, 500

# Handle all HTTP errors globally
@app.errorhandler(HTTPException)
def handle_http_error(e):
    return jsonify({"error": f"HTTP Error: {e.name} - {e.description}"}), e.code

# Add API resource
api.add_resource(Predict, "/predict")

# Start API with auto-reload with code changes
if __name__ == "__main__":
    app.run(debug=True)
