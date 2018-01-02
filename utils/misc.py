from morph import unflatten
import flatten_dict
def flatten(indict):
    return flatten_dict.flatten(indict, lambda k1, k2: k2 if k1 is None else k1 + '.' + k2)
