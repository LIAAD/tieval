
# constants representing the start point and end point of an interval
_START = 0
_END = 1

# Map relations to unique names.
_SETTLE_RELATION = {
    'OVERLAP': 'OVERLAP',
    'BEGINS': 'BEGINS',
    'BEFORE': 'BEFORE',
    'b': 'BEFORE',
    'CONTAINS': 'INCLUDES',
    'IDENTITY': 'SIMULTANEOUS',
    'EQUAL': 'SIMULTANEOUS',
    'AFTER': 'AFTER',
    'a': 'AFTER',
    'BEGINS-ON': 'BEGINS-ON',
    'SIMULTANEOUS': 'SIMULTANEOUS',
    's': 'SIMULTANEOUS',
    'INCLUDES': 'INCLUDES',
    'i': 'INCLUDES',
    'DURING': 'SIMULTANEOUS',
    'ENDS-ON': 'ENDS-ON',
    'BEGUN_BY': 'BEGUN_BY',
    'ENDED_BY': 'ENDED_BY',
    'DURING_INV': 'SIMULTANEOUS',
    'ENDS': 'ENDS',
    'IS_INCLUDED': 'IS_INCLUDED',
    'ii': 'IS_INCLUDED',
    'IBEFORE': 'IBEFORE',
    'IAFTER': 'IAFTER',
    'VAGUE': 'VAGUE',
    'v': 'VAGUE',
    'BEFORE-OR-OVERLAP': 'BEFORE-OR-OVERLAP',
    'OVERLAP-OR-AFTER': 'OVERLAP-OR-AFTER'
}

_INTERVAL_RELATIONS = list(_SETTLE_RELATION.keys())

# Mapping from interval relation names to point relations.
# For example, BEFORE means that the first interval's end is before the second interval's start
_INTERVAL_TO_POINT = {
    "BEFORE": [(_END, "<", _START)],
    "AFTER": [(_START, '>', _END)],
    "IBEFORE": [(_END, "=", _START)],
    "IAFTER": [(_START, "=", _END)],
    "INCLUDES": [(_START, "<", _START), (_END, '>', _END)],
    "IS_INCLUDED": [(_START, '>', _START), (_END, "<", _END)],
    "BEGINS-ON": [(_START, "=", _START)],
    "ENDS-ON": [(_END, "=", _END)],
    "BEGINS": [(_START, "=", _START), (_END, "<", _END)],
    "BEGUN_BY": [(_START, "=", _START), (_END, '>', _END)],
    "ENDS": [(_START, '>', _START), (_END, "=", _END)],
    "ENDED_BY": [(_START, "<", _START), (_END, "=", _END)],
    "SIMULTANEOUS": [(_START, "=", _START), (_END, "=", _END)],
    "OVERLAP": [(_START, "<", _END), (_END, '>', _START)],
    "VAGUE": [(_START, None, _START), (_START, None, _END), (_END, None, _START), (_END, None, _END)],
    'BEFORE-OR-OVERLAP': [(_START, '<', _START), (_END, '<', _END)],
    'OVERLAP-OR-AFTER': [(_START, '>', _START), (_END, '>', _END)]
}

_INTERVAL_TO_POINT_COMPLETE = {
    "BEFORE": [
        (_START, "<", _START),
        (_START, "<", _END),
        (_END, "<", _START),
        (_END, "<", _END)
    ],
    "AFTER": [
        (_START, ">", _START),
        (_START, ">", _END),
        (_END, ">", _START),
        (_END, ">", _END)
    ],
    "IBEFORE": [
        (_START, "<", _START),
        (_START, "=", _END),
        (_END, "<", _START),
        (_END, "<", _END)
    ],
    "IAFTER": [
        (_START, ">", _START),
        (_START, "=", _END),
        (_END, ">", _START),
        (_END, ">", _END)
    ],
    "INCLUDES": [
        (_START, "<", _START),
        (_START, "<", _END),
        (_END, ">", _START),
        (_END, ">", _END)
    ],
    "IS_INCLUDED": [
        (_START, ">", _START),
        (_START, "<", _END),
        (_END, ">", _START),
        (_END, "<", _END)
    ],
    "BEGINS-ON": [
        (_START, "=", _START),
        (_START, "<", _END),
        (_END, ">", _START),
        (_END, None, _END)
    ],
    "ENDS-ON": [
        (_START, None, _START),
        (_START, "<", _END),
        (_END, ">", _START),
        (_END, "=", _END)
    ],
    "BEGINS": [
        (_START, "=", _START),
        (_START, "<", _END),
        (_END, ">", _START),
        (_END, "<", _END)
    ],
    "BEGUN_BY": [
        (_START, "=", _START),
        (_START, "<", _END),
        (_END, ">", _START),
        (_END, ">", _END)
    ],
    "ENDS": [
        (_START, ">", _START),
        (_START, "<", _END),
        (_END, ">", _START),
        (_END, "=", _END)
    ],
    "ENDED_BY": [
        (_START, "<", _START),
        (_START, "<", _END),
        (_END, ">", _START),
        (_END, "=", _END)
    ],

    "SIMULTANEOUS": [
        (_START, "=", _START),
        (_START, "<", _END),
        (_END, ">", _START),
        (_END, "=", _END)
    ],
    "OVERLAP": [
        (_START, "<", _START),
        (_START, "<", _END),
        (_END, ">", _START),
        (_END, "<", _END)
    ],
    "VAGUE": [
        (_START, None, _START),
        (_START, None, _END),
        (_END, None, _START),
        (_END, None, _END)
    ],
    'BEFORE-OR-OVERLAP': [
        (_START, '<', _START),
        (_START, '<', _END),
        (_END, None, _START),
        (_END, '<', _END)
    ],
    'OVERLAP-OR-AFTER': [
        (_START, '>', _START),
        (_START, None, _END),
        (_END, '>', _START),
        (_END, '>', _END)
    ]
}

_POINT_RELATIONS = list(_INTERVAL_TO_POINT.values()) + \
                   list(_INTERVAL_TO_POINT_COMPLETE.values())

# transitivity table for point relations
_POINT_TRANSITIONS = {
    '<': {'<': '<', '=': '<', '>': None},
    '=': {'<': '<', '=': '=', '>': '>'},
    '>': {'>': '>', '=': '>', '<': None}
}

_INVERSE_POINT_RELATION = {
    '<': '>',
    '>': '<',
    '=': '='
}

_INVERSE_INTERVAL_RELATION = {
    'AFTER': 'BEFORE',
    'BEFORE': 'AFTER',
    'BEGINS': 'BEGUN_BY',
    'BEGINS-ON': 'BEGINS-ON',
    'BEGUN_BY': 'BEGINS',
    'ENDED_BY': 'ENDS',
    'ENDS': 'ENDED_BY',
    'ENDS-ON': 'ENDS-ON',
    'IAFTER': 'IBEFORE',
    'IBEFORE': 'IAFTER',
    'INCLUDES': 'IS_INCLUDED',
    'IS_INCLUDED': 'INCLUDES',
    'SIMULTANEOUS': 'SIMULTANEOUS',
    'OVERLAP': 'OVERLAP',
    'VAGUE': 'VAGUE'
}
