file = open("effective_reproductive_local.csv")
fout = open("effective_reproductive_local_2.csv","w")

for i in range(165):
    line = file.readline()
    if line[0] != "#":
        fout.write(line[0:11])
        fout.write('\n')