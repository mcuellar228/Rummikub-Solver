from Backend.Models.Tile import Tile
from Backend.Models.TileSet import TileSet
import copy
from collections import defaultdict, Counter

class Board:

  def __init__(self, tiles: list[Tile]):
    self.tiles = tiles
    self.valid = False
    self.valid_states = []
    self.tile_counts = defaultdict(Counter)
    for tile in tiles:
      self.tile_counts[tile.number][tile.color] += 1

  def getCurrentSetsTileCount(self, currentTotalSets: list[TileSet]):
    count = 0
    for tileSet in currentTotalSets:
      count += len(tileSet.tiles)
    return count

  def _sortCurrentSets(self, currentSets: list[TileSet]):
    currentSets.sort(key=lambda set: (set.tiles[0].number, set.tiles[0].color))

  def _convertValidStatesToStrings(self, tileSets: list[TileSet]):
    index_str = ""
    for i in range(len(tileSets)):
      for tile in tileSets[i].tiles:
        index_str += f"{tile.number}-{tile.color} "
      if i != len(tileSets)-1:
        index_str += "| "
    return index_str.strip()

  def _doesTileHaveValidNeighbors(self, tile):
    colors = {'black', 'orange', 'blue', 'red'}

    other_colors = colors - {tile.color}
    available_colors_for_number = set(self.tile_counts[tile.number].keys())
    matching_colors = available_colors_for_number & other_colors

    if len(matching_colors) >= 2:
        return True

    # Check if tile can form a RUN (consecutive numbers, same color)
    color = tile.color
    number = tile.number

    # Helper function to check if a number has tiles in the target color
    def has_tile_in_color(num):
        return self.tile_counts[num][color] > 0

    # Check all possible run positions for this tile
    run_possibilities = [
        has_tile_in_color(number + 1) and has_tile_in_color(number + 2),
        has_tile_in_color(number - 1) and has_tile_in_color(number + 1),
        has_tile_in_color(number - 2) and has_tile_in_color(number - 1)
    ]

    return any(run_possibilities)

  def _passesBasicValidCriteria(self):
    if len(self.tiles) < 3:
        return False
    for tile in self.tiles:
        if not self._doesTileHaveValidNeighbors(tile):
            return False
    return True

  def createValidBoards(self):
    if self._passesBasicValidCriteria():
      for i in range(len(self.tiles)):
        self.boardRecursion(i, self.tiles[i], TileSet([]), [], set())
    return self.valid_states

  def boardRecursion(self, index, currentTile: Tile, currentSet: TileSet, currentTotalSets: list[TileSet], usedIndices: set[int]):
    if len(self.valid_states) < 1:

      # if there are two tiles in the current set and they can't form a valid set return
      if len(currentSet.tiles) == 2 and currentSet.tiles[0].color != currentSet.tiles[1].color and currentSet.tiles[0].number != currentSet.tiles[1].number:
        return

      # check set to see if we're looking at the same tile twice
      if index in usedIndices:
        return

      usedIndices.add(index)

      # if all tiles are accounted for in currentTotalSets then we add currentaTotal to valid states and return
      if self.getCurrentSetsTileCount(currentTotalSets) == len(self.tiles):
        self._sortCurrentSets(currentTotalSets)
        self.valid_states = currentTotalSets
        return

      # Add the current tile to the current tile set
      currentSet.addTile(currentTile)

      # if the current set is valid and will constitute the last tiles we need to look at we add it to valid sates and return
      if self.getCurrentSetsTileCount(currentTotalSets) + len(currentSet.tiles) == len(self.tiles) and currentSet.is_valid_set():
        currentSet.sort_by_number_and_color()
        currentTotalSets.append(currentSet)
        self._sortCurrentSets(currentTotalSets)
        self.valid_states = currentTotalSets
        return

      # If there are more than 2 tiles in current set and it is not valid we return
      if len(currentSet.tiles) > 2 and not currentSet.is_valid_set():
        return

      # if there we have started to look at tiles we've already seen then return
      if self.getCurrentSetsTileCount(currentTotalSets) + len(currentSet.tiles) > len(self.tiles):
        return

      # Loop through remaining tiles, and either add to current set or start new set with the tile and make recursive call in both cases
      for i in range(len(self.tiles)):

        # If previous bases cases are not met then we make shallow copies of the total sets and current set
        currentTotalSetsCopy = copy.deepcopy(currentTotalSets)
        currentSetCopy = copy.deepcopy(currentSet)
        usedIndicesCopy = copy.deepcopy(usedIndices)
        self.boardRecursion(i, self.tiles[i], currentSetCopy, currentTotalSetsCopy, usedIndicesCopy)

        # Also make shallow copies for the second recursion call
        currentSetCopy2 = copy.deepcopy(currentSet)
        if len(currentSetCopy2.tiles) > 2 and currentSetCopy2.is_valid_set():
          usedIndicesCopy2 = copy.deepcopy(usedIndices)
          currentTotalSetsCopy2 = copy.deepcopy(currentTotalSets)
          currentSetCopy2.sort_by_number_and_color()
          currentTotalSetsCopy2.append(currentSetCopy2)
          self.boardRecursion(i, self.tiles[i], TileSet([]), currentTotalSetsCopy2, usedIndicesCopy2)