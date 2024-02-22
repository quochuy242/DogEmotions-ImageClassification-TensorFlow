import data.preprocessing as p

a = tuple([p.one_hot_encoding('angry') for _ in range(10)])
print(a)