from Models.Tile import Tile
from collections import Counter
from typing import List

class TileSet:
  def __init__(self, tiles: List[Tile]):
    self.tiles = tiles
    self._update_stats()

  def _update_stats(self):
    self.tile_num_counts = Counter(tile.number for tile in self.tiles)
    self.tile_color_counts = Counter(tile.color for tile in self.tiles)

    if self.tiles:
        self.max_num = max(self.tile_num_counts.keys())
        self.min_num = min(self.tile_num_counts.keys())
    else:
        self.max_num = None
        self.min_num = None

  def addTile(self, tile: Tile):
    self.tiles.append(tile)
    self.tile_num_counts[tile.number] += 1
    self.tile_color_counts[tile.color] += 1
    if self.max_num is None or tile.number > self.max_num:
        self.max_num = tile.number
    if self.min_num is None or tile.number < self.min_num:
        self.min_num = tile.number

  def to_dict(self):
    return {
      'tiles': [tile.__dict__ for tile in self.tiles]
  }

  def sort_by_number_and_color(self):
    self.tiles.sort(key=lambda tile: (tile.number, tile.color))

  def is_valid_set(self):
    return self._is_valid_run() or self._is_valid_group()

  def _is_valid_run(self):
    run_length = self.max_num-self.min_num+1
    max_number_repetitions = max(self.tile_num_counts.values())
    return len(self.tiles) > 2 and len(self.tile_color_counts) == 1 and len(self.tiles) == run_length and max_number_repetitions == 1

  def _is_valid_group(self):
    return (len(self.tiles) == 3 or len(self.tiles) == 4) and len(self.tile_color_counts) == len(self.tiles) and self.max_num == self.min_num