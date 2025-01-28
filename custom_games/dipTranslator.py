from diplomacy import Map as d_map

class DipTranslator:
    def __init__(self):
        self.loc_to_idx: dict[str, int]
        t_locs = list(d_map.locs).sort()
        for i in range(len(t_locs)):
            self.loc_to_idx[str(t_locs[i]).lower()] = i

        self.power_to_idx: dict[str, int]
        t_pow_name = list(d_map.pow_name.keys()).sort()
        for i in range(len(t+t_pow_name)):
            self.power_to_idx[str(t_pow_name[i]).lower()] = i

        self.keyword_to_idx: dict[str, int]
        t_keywords = list(d_map.keywords.values()).sort()
        for i in range(len(t_keywords)):
            if t_keywords[i].isspace() \
                or len(t_keywords[i]) == 0 \
                or t_keywords[i] in self.keyword_to_idx:
                continue
            self.keyword_to_idx[str(t_keywords[i]).lower()] = i

    def get_loc_idx_from_loc(self, loc: str) -> int | None:
        return self.loc_to_idx.get(loc.lower())

    def get_power_idx_from_power(self, power: str) -> int | None:
        return self.power_to_idx.get(power.lower())
    
    def _decode_raw_unit_text(self, raw_unit_text: str) -> tuple[int, int] | None:
        """Decode raw_unit_text of format 'unit_type loc' into a tuple of two ints
        - [0]: unit type. Value either 1 (A army), or 2 (F fleet).
        - [1]: loc idx

        Returns None if the format is incorrect. If unit type is incorrect, automatically assume it to be A army.
        """
        unit_type = 2 if raw_unit_text.split(' ')[0].lower() == 'f' else 1
        try:
            loc_idx = self.get_loc_idx_from_loc(raw_unit_text.split(' ')[1])
            if loc_idx == None:
                return None
        except IndexError: 
            return None
        return tuple(unit_type, loc_idx)
    
    def translate_game_state_to_matrix(self, game_state: dict, num_loc, num_pow, num_stats) -> list[list[list[int]]]:
        """
        Translate a game state into 3D matrix. This game state is from diplomacy.Game.get_state(), not DipState itself
        - x: loc
        - y: pow
        - z: stats (0: unit pos, 1: center pos, 2: influence pos)

        Each cell has value of either 0 or 1.
        For unit pos, value is either 0 (no unit), 1 (A army), or 2 (F fleet)
        """
        mtx = [[[0] * num_stats] * num_pow] * num_loc

        # Update unit pos
        for raw_pow_name, raw_units in game_state["units"]:
            raw_units: list[str]
            pow_idx = self.get_power_idx_from_power(raw_pow_name)
            if pow_idx == None: continue
            for raw_unit in raw_units:
                unit_type, loc_idx = self._decode_raw_unit_text(raw_unit)
                mtx[loc_idx][pow_idx][0] = unit_type

        # Update center pos
        for raw_pow_name, raw_locs in game_state["centers"]:
            raw_locs: list[str]
            pow_idx = self.get_power_idx_from_power(raw_pow_name)
            if pow_idx == None: continue
            for loc in raw_locs:
                loc_idx = self.get_loc_idx_from_loc(loc)
                if loc_idx == None: continue
                mtx[loc_idx][pow_idx][1] = 1
        
        # Update influence pos
        for raw_pow_name, raw_locs in game_state["influence"]:
            raw_locs: list[str]
            pow_idx = self.get_power_idx_from_power(raw_pow_name)
            if pow_idx == None: continue
            for loc in raw_locs:
                loc_idx = self.get_loc_idx_from_loc(loc)
                if loc_idx == None: continue
                mtx[loc_idx][pow_idx][1] = 1

        return mtx