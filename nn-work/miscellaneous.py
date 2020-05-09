class Miscellaneous():
    def printcmatrix(cmatrix):
        p = "positive"
        n = "negative"
        cmatrix.insert(0, ["", p, n])
        cmatrix[1].insert(0, p)
        cmatrix[2].insert(0, n)
        s = [[str(e) for e in row] for row in cmatrix]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = "\t".join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        print("\n".join(table))