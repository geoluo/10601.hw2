import sys
import csv
import math


class Node:
    def __init__(self, divide_by):
        self.divide_by = divide_by
        self.dividing_value = []
        self.divided_node = []
        self.is_leaf = False
        self.predict = ""


def parse_data(i_file_name):
    with open(i_file_name, 'r') as i_file:
        data = list(csv.reader(i_file))
        names_of_attribute = data[0]
        parsed_data = {}
        for i, name in enumerate(names_of_attribute):
            parsed_data[name] = [x[i] for x in data[1:]]
        i_file.close()
        return parsed_data, names_of_attribute


def entropy(data_set):
    entropy1 = 0
    n = len(data_set)
    a = {}
    for x in data_set:
        if x in a:
            a[x] += 1
        else:
            a[x] = 1
    for key in a:
        p = 1.0 * a[key] / n
        entropy1 -= p * math.log(p, 2)
    return entropy1


def information_gain(parsed_data, attribute1, res):
    r = parsed_data[res]
    entropy1 = entropy(r)
    entropy2 = 0
    ax = parsed_data[attribute1]
    divided_list = {}
    for i, x in enumerate(ax):
        if x in divided_list:
            divided_list[x].append(r[i])
        else:
            divided_list[x] = [r[i]]
    for key in divided_list:
        s = divided_list[key]
        entropy2 += entropy(s) * len(s) / len(r)
    # print(entropy1 - entropy2, attribute1, res)
    return entropy1 - entropy2


def divide(parsed_data, attributes, remain_levels, depth=1, all_available_values=[]):
    # print("pd")
    # print(parsed_data)
    if depth == 1:
        tmp = {}
        for x in parsed_data[attributes[-1]]:
            if x in tmp:
                tmp[x] += 1
            else:
                tmp[x] = 1
        k = [x for x in tmp]
        k.sort()
        stdout = "["
        for x in k:
            stdout = stdout + str(tmp[x]) + " " + x + " /"
        stdout = stdout[:-2] + "]"
        print(stdout)
        all_available_values = k
    if remain_levels >= len(attributes):
        remain_levels = len(attributes) - 1
    if remain_levels == 0:
        r = Node(attributes[-1])
        r.is_leaf = True
        tmp = parsed_data[attributes[-1]]
        tmp_map = {}
        for x in tmp:
            if x in tmp_map:
                tmp_map[x] += 1
            else:
                tmp_map[x] = 1
        c_max = 0
        c_name = ""
        for key in tmp_map:
            if tmp_map[key] > c_max:
                c_max = tmp_map[key]
                c_name = key
        r.predict = c_name
        return r
    tmp = parsed_data[attributes[-1]]
    if max(tmp) == min(tmp):
        r = Node(attributes[-1])
        r.is_leaf = True
        r.predict = tmp[0]
        return r
    c_max = 0
    c_name = ""
    # print(parsed_data)
    for name in attributes[: -1]:
        ig = information_gain(parsed_data, name, attributes[-1])
        if ig > c_max:
            c_max = ig
            c_name = name
    next_attributes = [x for x in attributes if x != c_name]
    divided_data = {}
    dividing_value = []

    if c_name == "":
        r = Node(attributes[-1])
        r.is_leaf = True
        tmp = parsed_data[attributes[-1]]
        tmp_map = {}
        for x in tmp:
            if x in tmp_map:
                tmp_map[x] += 1
            else:
                tmp_map[x] = 1
        c_max = 0
        c_name = ""
        for key in tmp_map:
            if tmp_map[key] > c_max:
                c_max = tmp_map[key]
                c_name = key
        r.predict = c_name
        return r

    for i, x in enumerate(parsed_data[c_name]):
        if x in divided_data:
            for key in parsed_data:
                if key != c_name:
                    divided_data[x][key].append(parsed_data[key][i])
        else:
            dividing_value.append(x)
            divided_data[x] = {}
            for key in parsed_data:
                if key != c_name:
                    divided_data[x][key] = [parsed_data[key][i]]
    r = Node(c_name)
    dividing_value.sort()
    r.dividing_value = dividing_value
    # r.divided_node = [divide(divided_data[x], next_attributes, remain_levels - 1, depth + 1) for x in dividing_value]
    for x_val in dividing_value:
        tmp = {}
        for i, y in enumerate(parsed_data[c_name]):
            if y == x_val:
                t2 = parsed_data[attributes[-1]][i]
                if t2 in tmp:
                    tmp[t2] += 1
                else:
                    tmp[t2] = 1
        k = [x for x in tmp]
        k.sort()
        stdout = "| " * depth + c_name + " = " + x_val + ": ["
        for x in all_available_values:
            if x in k:
                stdout = stdout + str(tmp[x]) + " " + x + " /"
            else:
                stdout = stdout + "0" + " " + x + " /"

        stdout = stdout[:-2] + "]"
        print(stdout)
        r.divided_node.append(divide(divided_data[x_val], next_attributes, remain_levels - 1, depth + 1, all_available_values))
    return r


def print_tree(root_node, length=0):
    if root_node.is_leaf:
        print(" " * length + str(root_node.divide_by))
    else:
        print(" " * length + str(root_node.divide_by))
        for x in root_node.divided_node:
            print_tree(x, length + 1)


def training():
    parsed_data, attributes = parse_data(train_input)
    root = divide(parsed_data, attributes, max_depth)
    return root


def forward(root, att_val_dict):
    if root.is_leaf:
        return root.predict
    else:
        for i, x in enumerate(root.dividing_value):
            if att_val_dict[root.divide_by] == x:
                return forward(root.divided_node[i], att_val_dict)
        return "error"


def output_labels(root, i_file_name, o_file_name):
    out = []
    with open(i_file_name, 'r') as i_file:
        data = list(csv.reader(i_file))
        attributes = data[0]
        for i, x in enumerate(data[1:]):
            tmp = {}
            for j, y in enumerate(attributes):
                tmp[y] = x[j]
            out.append(forward(root, tmp))
    with open(o_file_name, "w+") as o_file:
        for line in out:
            o_file.write(line + '\n')


def output_metrics(root, train_file, test_file, o_file_name):
    in_tr_data, in_tr_attr = parse_data(train_file)
    in_tr = in_tr_data[in_tr_attr[-1]]
    in_te_data, in_te_attr = parse_data(test_file)
    in_te = in_te_data[in_te_attr[-1]]
    out_tr = []
    out_te = []
    with open(train_file, 'r') as tr_file:
        data = list(csv.reader(tr_file))
        attributes = data[0]
        for i, x in enumerate(data[1:]):
            tmp = {}
            for j, y in enumerate(attributes):
                tmp[y] = x[j]
            out_tr.append(forward(root, tmp))
    with open(test_file, 'r') as te_file:
        data = list(csv.reader(te_file))
        attributes = data[0]
        for i, x in enumerate(data[1:]):
            tmp = {}
            for j, y in enumerate(attributes):
                tmp[y] = x[j]
            out_te.append(forward(root, tmp))
    e1 = 0
    e2 = 0
    for i in range(len(out_tr)):
        if in_tr[i] != out_tr[i]:
            e1 += 1.0/ len(out_tr)
    for i in range(len(out_te)):
        if in_te[i] != out_te[i]:
            e2 += 1.0/ len(out_te)
    with open(o_file_name, "w+") as o_file:
        o_file.write("error(train): " + str(e1) + "\n")
        o_file.write("error(test): " + str(e2) + "\n")
def decision_tree():
    parsed_data, attributes = parse_data(train_input)
    m_depth = min(max_depth, len(parsed_data[attributes[-1]]))
    root = divide(parsed_data, attributes, m_depth)
    #root = training()
    output_labels(root, train_input, train_out)
    output_labels(root, test_input, test_out)
    output_metrics(root,train_input,test_input,metric_out)


if __name__ == '__main__':
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    max_depth = int(sys.argv[3])
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metric_out = sys.argv[6]
    decision_tree()

