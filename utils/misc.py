from morph import unflatten
import flatten_dict
def dot_reducer(k1, k2):
    if k1 is None:
        return k2
    else:
        return k1+'.'+k2
def flatten(indict):
    return flatten_dict.flatten(indict, lambda k1, k2: k2 if k1 is None else k1 + '.' + k2)
