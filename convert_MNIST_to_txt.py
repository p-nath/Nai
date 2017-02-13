import os
import struct

DIGIT_WIDTH=28
DIGIT_HEIGHT=28

def convertdata(filename, fout_name, n, width, height):
  DATUM_WIDTH=width
  DATUM_HEIGHT=height
  fin = readrawdata(filename, n)
  with open(fout_name, "w") as fout:
    for k in xrange(n):
      for i in xrange(28):
        for j in xrange(28):
          fout.write(str(fin[k][i][j]))
        fout.write("\n")

def readrawdata(filename, n):
  "Opens a file or reads it from the zip archive data.zip"
  if(os.path.exists(filename)): 
    count = 0;
    with open(filename) as f:
      magic_num = struct.unpack(">I", f.read(4))[0]
      total = struct.unpack(">I", f.read(4))[0]
      rows = struct.unpack(">I", f.read(4))[0]
      cols = struct.unpack(">I", f.read(4))[0]
      #print magic_num, total,rows,cols
      l = []
      for k in xrange(n):
        digit = []
        for i in range(DIGIT_HEIGHT):
          row = []
          for j in xrange(DIGIT_WIDTH):
            if (int(struct.unpack(">B", f.read(1))[0]) != 0):
              row.append("+")
            else:
              row.append(" ")
          digit.append(row)
        l.append(digit)
      return l

def convertlabels(filename, fout_name, n):
  fin = readrawlabels(filename, n)
  labels = []
  with open(fout_name, 'w') as fout:
    for i in xrange(len(fin)):
      fout.write(str(fin[i]))
  
def readrawlabels(filename, n) :
  if(os.path.exists(filename)): 
    with open(filename) as f:
      magic_num = struct.unpack(">I", f.read(4))[0]
      total = struct.unpack(">I", f.read(4))[0]
      #print magic_num, total
      l = []
      for k in xrange(n):
        l.append(struct.unpack(">B", f.read(1))[0])
        l.append('\n')
      return l



convertlabels("digitdata_full/traininglabels", "data/traininglabels", 60000)
convertdata("digitdata_full/trainingimages", "data/trainingimages", 60000, DIGIT_WIDTH, DIGIT_HEIGHT)
convertdata("digitdata_full/testimages", "data/testimages", 10000, DIGIT_WIDTH, DIGIT_HEIGHT)
convertlabels("digitdata_full/testlabels", "data/testlabels", 10000)




