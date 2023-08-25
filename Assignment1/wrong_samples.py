import jsonlines, numpy

filename = 'wrong_samples_10.txt' # give a name
output_items = [] # list of your 10 instances in the format of a dictionary {'review': <review text>, 'label': <gold label>, 'predicted': <predicted label>}



with jsonlines.open(filename, mode='w') as writer:
    for item in output_items:
        writer.write(item)
    ##
##

