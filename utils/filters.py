def rcs_scale(value: float) -> float:

    if value >= 10.0:
        return 1.0
    elif value <= -10.0:
        return 0.0
    else:
        return (value + 10.0) / 20.0 

def vr_scale(value: float) -> float:

    if value >= 7.50:
        return 1.0
    elif value <= -7.50:
        return 0.0
    else:
        return (value + 7.5) / 15.0