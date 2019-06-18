maxMpg = 46.6
minMpg = 10


def scale(value, max, min):
    return (value - min) / (max - min)

def reverseScale(value, max, min):
    return value * (max - min) + min

def scaleMPG(mpg):
    return scale(mpg, maxMpg, minMpg)

def revertScaleMPG(scaledMpg):
    return reverseScale(mpg, maxMpg, minMpg)

print(scaleMPG(2))