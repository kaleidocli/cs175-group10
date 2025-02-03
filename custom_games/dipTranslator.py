from diplomacy import Map as d_map

class DipTranslator:
    def __init__(self):
        self.dMap = d_map()
        self._loc_to_idx: dict[str, int] = dict()
        t_locs = sorted(list(self.dMap.locs))
        for i in range(len(t_locs)):
            self._loc_to_idx[str(t_locs[i]).lower()] = i

        self._power_to_idx: dict[str, int] = dict()
        self._powers = sorted(list(self.dMap.pow_name.keys()))
        for i in range(len(self._powers)):
            self._power_to_idx[str(self._powers[i]).lower()] = i

        self._keyword_to_idx: dict[str, int] = dict()
        t_keywords = sorted(list(set(i for i in self.dMap.keywords.values() if len(i) > 0)))
        for i in range(len(t_keywords)):
            if t_keywords[i].isspace() \
                or len(t_keywords[i]) == 0 \
                or t_keywords[i] in self._keyword_to_idx:
                continue
            self._keyword_to_idx[str(t_keywords[i]).lower()] = i

        self._action_id_to_action: dict[int, list[str]] = dict()

        self.UNIT_TYPE_COUNT = 3        # [None, A, S]
        self.LOC_COUNT = self._get_loc_count() + 1      # [None, Loc1,..]
        self.ACTION_TYPE_COUNT = 6                      # [None, Hold, Move, Support, Convoy, Disband]
        self.POWER_COUNT = self._get_power_count()

    def get_loc_idx_from_loc(self, loc: str) -> int | None:
        return self._loc_to_idx.get(loc.lower())

    def get_power_idx_from_power(self, power: str) -> int | None:
        return self._power_to_idx.get(power.lower())
    
    def get_power_from_power_idx(self, power_idx: int) -> str | None:
        return self._powers[power_idx] if power_idx < len(self._powers) else None
    
    def _get_power_count(self) -> int:
        return len(self._power_to_idx)
    
    def _get_loc_count(self) -> int:
        return len(self._loc_to_idx)
    
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
        return tuple({unit_type, loc_idx})
    
    def serialize_action_to_int(self, action: list[str]) -> int:
        """Action is action (a ser of orders) from diplomacy.py"""
        result = 0
        for order in action:
            result += self.serialize_order_to_int(order)
        self._action_id_to_action.update({result: action})
        return result
    
    def deserialize_action_id_by_cache(self, action_id: int) -> list[str] | None:
        return self._action_id_to_action.get(action_id)

    def serialize_order_to_int(self, order: str) -> int:
        result = 0
        for word in order.split(' '):       # we still record space for unknown words
            if word.lower() == "a":
                result = 1 if result == 0 else result * 10 + 1
            elif word.lower() == "s":
                result = 2 if result == 0 else result * 10 + 2
            elif word.lower() == "h":
                result = 3 if result == 0 else result * 10 + 3
            elif word.lower() == "-":
                result = 4 if result == 0 else result * 10 + 4
            elif word.lower() == "f":
                result = 5 if result == 0 else result * 10 + 5
            elif word.lower() in self._keyword_to_idx:
                result = (result + self._keyword_to_idx.get(word)) * 100
            elif word.lower() in self._power_to_idx:
                result = (result + self.get_power_idx_from_power(word)) * 100
            else:
                result = 100 if result == 0 else result * 100

        if result % 100 == 0:
            result %= 100
        return result
        
    
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
        fail = 0

        # Update unit pos
        for raw_pow_name, raw_units in game_state["units"].items():
            raw_units: list[str]
            pow_idx = self.get_power_idx_from_power(raw_pow_name)
            if pow_idx == None: 
                fail += 1
                continue
            for raw_unit in raw_units:
                unit_type, loc_idx = self._decode_raw_unit_text(raw_unit)
                mtx[loc_idx][pow_idx][0] = unit_type

        # Update center pos
        for raw_pow_name, raw_locs in game_state["centers"].items():
            raw_locs: list[str]
            pow_idx = self.get_power_idx_from_power(raw_pow_name)
            if pow_idx == None: 
                fail += 1
                continue
            for loc in raw_locs:
                loc_idx = self.get_loc_idx_from_loc(loc)
                if loc_idx == None: 
                    fail += 1
                    continue
                mtx[loc_idx][pow_idx][1] = 1
        
        # Update influence pos
        for raw_pow_name, raw_locs in game_state["influence"].items():
            raw_locs: list[str]
            pow_idx = self.get_power_idx_from_power(raw_pow_name)
            if pow_idx == None: 
                fail += 1
                continue
            for loc in raw_locs:
                loc_idx = self.get_loc_idx_from_loc(loc)
                if loc_idx == None: 
                    fail += 1
                    continue
                mtx[loc_idx][pow_idx][1] = 1

        print(f"[dipTranslator]\t\tTranslated {num_loc * num_pow * num_stats} cells with {fail} failures.")

        return mtx