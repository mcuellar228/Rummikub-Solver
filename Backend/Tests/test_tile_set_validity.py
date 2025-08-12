import pytest
from Models.Tile import Tile
from Models.TileSet import TileSet

def test_good_group_1_returns_valid_group():
  tile_set = TileSet([
    Tile(1, "red"),
    Tile(1, "blue"),
    Tile(1, "black")
  ])
  assert tile_set._is_valid_group()
  assert tile_set.is_valid_set()

def test_good_group_2_returns_valid_group():
  tile_set = TileSet([
    Tile(1, "red"),
    Tile(1, "blue"),
    Tile(1, "black"),
    Tile(1, "orange")
  ])
  assert tile_set._is_valid_group()
  assert tile_set.is_valid_set()

def test_bad_group_1_returns_not_valid_group():
  tile_set = TileSet([
    Tile(1, "red"),
    Tile(1, "blue"),
    Tile(1, "black"),
    Tile(1, "black")
  ])
  assert not tile_set._is_valid_group()
  assert not tile_set.is_valid_set()

def test_bad_group_2_returns_not_valid_group():
  tile_set = TileSet([
    Tile(1, "red"),
    Tile(1, "black"),
    Tile(1, "black")
  ])
  assert not tile_set._is_valid_group()
  assert not tile_set.is_valid_set()

def test_bad_group_3_returns_not_valid_group():
  tile_set = TileSet([
    Tile(1, "red"),
    Tile(1, "black"),
  ])
  assert not tile_set._is_valid_group()
  assert not tile_set.is_valid_set()

def test_bad_group_4_returns_not_valid_group():
  tile_set = TileSet([
    Tile(1, "red"),
    Tile(1, "black"),
    Tile(2, "blue")
  ])
  assert not tile_set._is_valid_group()
  assert not tile_set.is_valid_set()

def test_good_run_1_returns_valid_run():
  tile_set = TileSet([
    Tile(1, "red"),
    Tile(2, "red"),
    Tile(3, "red")
  ])
  assert tile_set._is_valid_run()
  assert tile_set.is_valid_set()

def test_good_run_2_returns_valid_run():
  tile_set = TileSet([
    Tile(6, "red"),
    Tile(2, "red"),
    Tile(3, "red"),
    Tile(4, "red"),
    Tile(5, "red")
  ])
  assert tile_set._is_valid_run()
  assert tile_set.is_valid_set()

def test_bad_run_1_returns_not_valid_run():
  tile_set = TileSet([
    Tile(1, "blue"),
    Tile(2, "blue"),
    Tile(3, "black"),
    Tile(4, "blue")
  ])
  assert not tile_set._is_valid_run()
  assert not tile_set.is_valid_set()

def test_bad_run_2_returns_not_valid_run():
  tile_set = TileSet([
    Tile(1, "blue"),
    Tile(2, "blue"),
    Tile(3, "blue"),
    Tile(4, "blue"),
    Tile(10, "blue"),
    Tile(5, "blue"),
    Tile(6, "blue"),
    Tile(7, "blue")
  ])
  assert not tile_set._is_valid_run()
  assert not tile_set.is_valid_set()


def test_bad_run_3_returns_not_valid_run():
  tile_set = TileSet([
    Tile(1, "blue"),
    Tile(2, "blue"),
  ])
  assert not tile_set._is_valid_run()
  assert not tile_set.is_valid_set()

def test_bad_set_1_returns_not_valid_run():
  tile_set = TileSet([
    Tile(1, "blue"),
    Tile(2, "blue"),
    Tile(3, "blue"),
    Tile(4, "red"),
    Tile(5, "red"),
    Tile(6, "red")
  ])
  assert not tile_set.is_valid_set()