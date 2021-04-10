def calcCJ(mural, x):
    count = 0
    if 'CJ' in mural:
        count = count + mural.count('CJ')
    return count*x

def calcJC(mural, y):
    count = 0
    if 'JC' in mural:
        count = count + mural.count('JC')
    return count*y

def calcTotal(x, y, mural):
    multaCJ = calcCJ(mural, x)
    multaJC = calcJC(mural, y)
    multa = multaCJ + multaJC
    return multa

def muralReplace(x, y, mural):
    aux = 1000
    opcao = ['C','J']
    for a in opcao:
        auxiliar = mural.replace('?',a)
        print(auxiliar)
        multa = calcTotal(x, y, auxiliar)
        if multa < aux:
            aux = multa
    return aux

num_entradas = int(input())

for i in range(num_entradas):

    entrada = input().split(' ')
    
    x = int(entrada[0])
    y = int(entrada[1])
    mural = entrada[2]
    if '?' in mural:
        multamin = muralReplace(x, y, mural)
        print('Case #{}:'.format(i+1), multamin)
    else:
        multa = calcTotal(x, y, mural)
        print('Case #{}:'.format(i+1), multa)
