import os
from shutil import copy
with open('evaluation_setup/fold1_evaluate.txt', 'rb') as f:
    count = 1
    
    for row in f:
        row = row.split()
#        print row


        base = 'audio/test/'
        if not os.path.exists(base+row[1]):
            os.makedirs(base+row[1])

        os.rename(row[0], base+row[1]+'/'+row[0].split('/')[1])
        print count
        count +=1 
