def make_z(treatment, target):
    y = target
    w = treatment
    z = y * w + (1 - y) * (1 - w)
    return z


def calc_uplift(prediction):
    uplift = 2 * prediction - 1
    return uplift
