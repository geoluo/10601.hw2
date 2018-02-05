import sys
import csv
import math


def inspect(i_file_name, o_file_name):
    entropy = 0
    error = 1
    with open(i_file_name,'r') as i_file:
        data = list(csv.reader(i_file))
        n = len(data) - 1
        r = [x[-1] for x in data][1:]
        a = {}
        for x in r:
            if x in a:
                a[x] += 1
            else:
                a[x] = 1

        for key in a:
            p = 1.0 * a[key] / n
            entropy -= p * math.log(p, 2)
            if 1 - p < error:
                error = 1 - p
        i_file.close()
    with open(o_file_name,"w+") as o_file:
        o_file.write('entropy: ' + str(entropy) + "\n")
        o_file.write('error: ' + str(error))
        o_file.close()


if __name__ == '__main__':
    i_file_name = sys.argv[1]
    o_file_name = sys.argv[2]
    inspect(i_file_name, o_file_name)