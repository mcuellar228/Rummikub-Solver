import os
from flask import Flask, request
from flask_cors import CORS
from Controllers import return_valid_board_states, process_image
from dotenv import load_dotenv

load_dotenv()
HOST = os.getenv('HOST')
PORT = os.getenv('PORT')

print(HOST,PORT)

app = Flask(__name__)
CORS(app)

@app.route("/tiles", methods=["POST"])
def tiles():
    if request.is_json:
      try:
        tiles = request.get_json()

        valid_board_sets = return_valid_board_states(tiles)

        return {"valid_board_sets": valid_board_sets}, 200

      except ValueError as e:
        return {"error": f"Invalid tile data: {str(e)}"}, 400
      except Exception as e:
        app.logger.error(f"Tiles processing error: {str(e)}")
        return {"error": "Internal server error"}, 500
    else:
      return {"error": "Request must be JSON"}, 400

@app.route("/image", methods=["POST"])
def image():
  try:
    if 'image' not in request.files:
        return {'error': 'No image file provided'}, 400

    image_file = request.files['image']

    if image_file.filename == '':
        return {'error': 'No image file selected'}, 400

    tiles = process_image(image_file)

    if not tiles:
        return {'error': 'No tiles detected in image'}, 400

    valid_board_sets = return_valid_board_states(tiles)

    return {'detected_tiles': tiles, 'valid_board_sets': valid_board_sets}, 200

  except ValueError as e:
    return {'error': f'Invalid input: {str(e)}'}, 400
  except Exception as e:
        # Log the full error for debugging
        app.logger.error(f"Image processing error: {str(e)}")
        return {'error': 'Internal server error'}, 500

@app.route('/test', methods=['GET'])
def test():
  return {"message":"Flask is working!"}, 200

if __name__ == '__main__':
  app.run(host=HOST, port=PORT, debug=True)
