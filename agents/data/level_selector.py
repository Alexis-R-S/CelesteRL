import json
from typing import Optional
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
json_data_path = BASE_DIR / "rooms.json"

class LevelSelector:
    def __init__(self):
        self.rooms_data = self._load_level_data()

    def get_level_pool(self, chapter: Optional[int]=None, side: Optional[int]=None):
        """Fetch the level pool for the given chapter and side
        Args:
            chapter: chapter selection. Leave to None to select all chapters
            side: side selection. Leave to None to select all sides
        Returns:
            Available rooms in format (chapter, side, room)
        """
        if chapter is None:
            return self._load_every_level()
        if side is None:
            return self._load_every_sides(chapter)
        return self._load_one_side(chapter, side)

    def _load_level_data(self):
        with open(json_data_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            # Convert data to exploitable dict with format:
            # {chapter_id: {side_id: [room_tags]}}
            chapters_rooms = {}

            for chapter_object in data["chapters"]:
                chap_id = chapter_object["id"]
                chapters_rooms[chap_id] = {}
                for side_object in chapter_object["sides"]:
                    side_id = side_object["side"]
                    chapters_rooms[chap_id][side_id] = side_object["rooms"]

            return chapters_rooms

    def _load_every_level(self):
        return sum([self._load_every_sides(chapter) for chapter in self.rooms_data.keys()], [])

    def _load_every_sides(self, chapter: int):
        return sum([self._load_one_side(chapter, side) for side in self.rooms_data[chapter].keys()], [])

    def _load_one_side(self, chapter: int, side: int):
        return [(chapter, side, room) for room in self.rooms_data[chapter][side]]