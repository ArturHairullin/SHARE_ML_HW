import csv
def vel(t, x, y):
    return (((60*60*x)/(1000*t))**2 + ((60*60*y)/(1000*t))**2)**0.5

def max_time(a):
    m = 0
    tmp = 0
    for s in a:
        if s[0] > 40:
            tmp+=s[1]
        else:
            if tmp > m:
                m = tmp
            tmp = 0
    if tmp > m:
        m = tmp
    return m

def difs(a):
    t = a[0]
    for s in a[1:]:
        yield vel(float(s[0])-float(t[0]), float(s[3])-float(t[3]), float(s[4])-float(t[4])), float(s[0])-float(t[0])
        t = s

def id_filter(ID,a):
    for s in a:
        if s[1] == ID:
            yield s
def ids(a):
    for s in a:
        yield s[1]
def gen_solution(path):
    a = []
    with open(path, newline='') as f:
        reader = csv.reader(f)
        a = list(reader)
    a = a[1:]
    IDs = set(ids(a))
    for ID in IDs:
        b = list(id_filter(ID,a))
        b.sort(key=lambda x: x[0])
        t = max_time(list(difs(b)))
        if not t < 1:
            yield ID
b = list(gen_solution('data.csv'))
b.sort()
with open('answer.txt','w') as f:
    f.write(b[0])
    for s in b[1:]:
        f.write('\n' + s)