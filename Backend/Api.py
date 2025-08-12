import os
from flask import Flask, request
from flask_cors import CORS
from Backend.Controllers import return_valid_board_states, process_image

HOST = os.environ.get('HOST')
PORT = os.environ.get('PORT')

app = Flask(__name__)
CORS(app)

@app.route("/tiles", methods=["POST"])
def tiles():
  if request.is_json:
    tiles = request.get_json()
    print(tiles)
    return return_valid_board_states(tiles)
  else:
    return {"error": "Request must be JSON"}, 400

@app.route("/image", methods=["POST"])
def image():
  tiles = process_image()
  return return_valid_board_states(tiles)

@app.route('/test', methods=['GET'])
def test():
  return {"message":"Flask is working!"}, 200

if __name__ == '__main__':
  app.run(host=HOST, port=PORT, debug=True)
