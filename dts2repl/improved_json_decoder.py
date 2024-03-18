import json
import re

class hexnum:
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return hex(self.value)

# code from json module, modified to support hexadecimal values
# license: MIT
class ImprovedJsonDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, *args, **kwargs)
        self.NUMBER_RE = re.compile(
    r'(-?(?:0|[1-9]\d*))(\.\d+)?([eE][-+]?\d+)?',
    (re.VERBOSE | re.MULTILINE | re.DOTALL))
        self.HEX_RE = re.compile(r'0x([0-9a-fA-F]+)?')
        self.scan_once = self._scan_once
    def _scan_once(self, string, idx):
        try:
            nextchar = string[idx]
        except IndexError:
            raise StopIteration(idx) from None

        if nextchar == '"':
            return self.parse_string(string, idx + 1, self.strict)
        elif nextchar == '{':
            return self.parse_object((string, idx + 1), self.strict,
                self._scan_once, self.object_hook, self.object_pairs_hook, self.memo)
        elif nextchar == '[':
            return self.parse_array((string, idx + 1), self._scan_once)
        elif nextchar == 'n' and string[idx:idx + 4] == 'null':
            return None, idx + 4
        elif nextchar == 't' and string[idx:idx + 4] == 'true':
            return True, idx + 4
        elif nextchar == 'f' and string[idx:idx + 5] == 'false':
            return False, idx + 5

        m = self.NUMBER_RE.match(string, idx)
        if m is not None:
            integer, frac, exp = m.groups()
            if frac or exp:
                res = self.parse_float(integer + (frac or '') + (exp or ''))
            else:
                if (string[idx:idx+2]  == "0x"):
                   m = self.HEX_RE.match(string, idx)
                   integer = m.groups()[0]
                   res = hexnum(self.parse_int(integer, 16))
                else:
                   res = self.parse_int(integer)
            return res, m.end()
        else:
            raise StopIteration(idx)

