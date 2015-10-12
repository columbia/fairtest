"""
Add percentages to timing results
"""

FILES = ['timing_discovery.csv', 'timing_testing.csv', 'timing_error.csv']

count = 0
for _file in FILES:
    inf = open(_file)
    data = inf.readlines()
    inf.close()
    
    outf = open(_file, 'w')
    outf.write('#set,train,test,ptrain,ptest\n')
    for i in range(1, len(data)):
        split = data[i].strip().split(',')
        print split
        dset = split[0]
        train = float(split[1])
        test = float(split[2])
        ptrain = 100*train/(train+test)
        ptest = 100*test/(train+test)
        outf.write('{},{},{},{:.2f},{:.2f},{},{}\n'.
                   format(dset, train, test, ptrain, ptest,
                          int(train + test), count))
        count += 1
    outf.close()
    count += 1