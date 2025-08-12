from Models.Board import Board
from Models.Tile import Tile
from flask import request
from Model.src.ComputerVision.DetectAndCutTiles import detectTiles

def return_valid_board_states(tiles):
  board_tiles = []
  for tile in tiles:
    board_tiles.append(Tile(tile['number'], tile['color']))
  state = (Board(board_tiles).createValidBoards())
  if len(state) > 0:
    json_data = [tile_set.to_dict() for tile_set in state]
    return json_data
  else:
    return {"message": "No valid states"}

def process_image():
  file_storage = request.files['image']
  result = detectTiles(file_storage)
  return result
