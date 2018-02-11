
from collections import defaultdict



s = [('yellow', 1), ('blue', 2), ('yellow', 3), ('blue', 4), ('red', 1)]
d = defaultdict(list)
for k, v in s:
    d[k].append(v)

print d.items()
f = [[1,2,3],[4,5,6]]
f = tuple(f)

print f[0]

d[tuple(f[0])].append(1)
d[tuple(f[1])].append(2)

print d[tuple(f[1])]