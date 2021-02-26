n_valores_vetor = int(input())
n_valores_pegos = int(input())
contador=0
vetor_entrada=[]
vetor_entrada_abs=[]
while (contador<n_valores_vetor):
    valor = int(input())
    vetor_entrada.append(valor)
    vetor_entrada_abs.append(abs(valor))
    contador=contador+1
contador=0
produto=1
modulo=1
while (contador<n_valores_pegos):
    posicao_max = vetor_entrada.index(max(vetor_entrada))
    valor_max_abs = vetor_entrada_abs.pop(posicao_max)
    valor_max = vetor_entrada.pop(posicao_max)
    produto = produto*valor_max
    modulo = modulo*valor_max_abs
    contador=contador+1
def myLog(x, b):
    if x < b:
        return 0  
    return 1 + myLog(x/b, b)

valor_grande = 1000000007
if produto<0:
    print(valor_grande+produto)
elif modulo>valor_grande:
    expoente = int(myLog(produto,10))
    aux = int(valor_grande - 7**(expoente/9))
    print(aux)
else:
    print(modulo)





