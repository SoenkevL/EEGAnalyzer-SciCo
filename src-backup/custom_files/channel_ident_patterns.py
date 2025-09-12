'''
This file is used to store channel identification patterns for the preprocessing viewer
'''
PATTERN_NAME = 'mysticalEntropyBDF'
MERGE_WITH_DEFAULT = False

def get_custom_pattern():
    if PATTERN_NAME == 'mysticalEntropyBDF':
        return bdf_pattern.copy()
    if PATTERN_NAME == 'anes':
        return anes_pattern.copy()
    return {}

bdf_pattern = {
    # Eyes: EXG1 (EOG left - bottom), EXG3 (EOG right - outside), EXG8 (EOG left - outside)
    'EOG': [
        r'^(EXG1|EXG3|EXG8)$',
        r'\bEOG\b',            # generic EOG fallback
        r'\b(VEOG|HEOG)\b',    # vertical/horizontal EOG
    ],

    # Muscle: EXG2 (EMG jaw)
    'EMG': [
        r'^EXG2$',
        r'\bEMG\b',            # generic EMG fallback
    ],

    # Heart: EXG6 (ECG)
    'ECG': [
        r'^EXG6$',
        r'\bECG\b',            # generic ECG
        r'\bEKG\b',            # common alternative spelling
    ],

    # Misc/ears: EXG4 (right ear lobe), EXG5 (left ear lobe)
    'MISC': [
        r'^(EXG4|EXG5)$',
    ],
}

anes_pattern = {
            'EOG': [
                r'.*Ref-?2.*',  # Matches anything containing 'Ref-0' or 'Ref0'
            ],
            'ECG': [
                r'.*[Ii][Nn].*',  # Matches anything containing 'In' or 'ln' (case insensitive)
            ]
        }