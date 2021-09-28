from collections import OrderedDict, namedtuple
from itertools import product

def generate_parameter_space(
    params: OrderedDict
) -> list:
    
    case = namedtuple('case', params.keys())
    cases = [ case(*values) for values in product(*params.values()) ]

    return cases
        
