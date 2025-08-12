class Tile:

  def __init__(self, number: int, color: str):
    self.number = number
    self.color = color

  def to_dict(self):
    return {
        'number': self.number,
        'color': self.color
    }