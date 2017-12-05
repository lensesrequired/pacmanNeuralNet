import layout

testLayouts = layout.loadLayouts( "goodmazes" )
for l, lay in enumerate(testLayouts):
  print l
  file = open("testLay" + str(l) + (".lay"), "w")
  lay = str(lay)
  newLay = ""
  for ch in lay:
    if ch == " ":
      newLay += "."
    else:
      newLay += ch
  file.write(newLay)
  print newLay