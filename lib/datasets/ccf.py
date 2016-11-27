#!/usr/bin/pyhon

# --------------------------------------------------------
# ccf competion
# Copyright (c) 2016 Peking University
# Licensed under The MIT License [see LICENSE for details]
# Written by coder-james
# --------------------------------------------------------
bases = ('__background__','head','top','down','shoes','hat','bag')
classes = ('__background__','woman', 'longhair', 'downjacket', 'pants', 'singleshoulder', 'coat', 'boots', 'man', 'shorthair','otherbag', 'hat', 'backpack', 'skirt', 'leathershoes', 'otherhair', 'sneakers', 'othertop', 'maxiskit', 'bag', 't-shirt', 'blouse', 'western', 'shorts', 'otherdown', 'othershoes', 'handbox', 'sandal', 'otherman', 'wallet')
colors = ('__background__','black', 'white', 'red', 'yellow', 'blue', 'green', 'purple', 'brown', 'gray', 'orange', 'multicolor', 'othercolor')

def getClasses(name):
  if name == "base":
    return bases
  elif name == "class":
    return classes
  elif name == "color":
    return colors
  else:
    print "function param should be 'base/class/color'"
    assert True == False
    return None
