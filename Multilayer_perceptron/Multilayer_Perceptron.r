'''
Name: Erick Correia Silva
Example of multilayer perceptron with iris dataset
'''

#carregando bibliotecas 
library(neuralnet) #blioteca para o multilayer perceptron
library(mltools)
library(data.table)
library(caret) #biblioteca para separação dos dados

#Padronizando as features
iris2 =  scale(iris[,1:4])
iris2 = as.data.frame(iris2)
#Adicionando a classe
iris2$Species = iris$Species
iris2

#dividindo os dados para treino e teste
set.seed(1234) #semente randomização 
particao = createDataPartition(1:dim(iris2)[1],p=.7)#separando 70% para treiono 
iristreino = iris2[particao$Resample1,]#treino 70%
iristeste = iris2[- particao$Resample1,]#teste 30%
#shape da separação 
dim(iristreino)
dim(iristeste)


#transformando a classe categorica em numerica usando one hot encoding
#cbind junta as os atributos com a classe para não perde-los
iristreino = cbind(iristreino[,1:4],one_hot(as.data.table( iristreino[,5])))
iristreino

#instanciano o modelo 
modelo = neuralnet( V1_setosa  + V1_versicolor  +  V1_virginica  ~ Sepal.Length + Sepal.Width +  Petal.Length + Petal.Width , iristreino, hidden=c(5,4))
#hidden=c(5,4) topologia da rede neural, 2 camadas ocultas, a primeira com 5 neuronios, a segunda com 4
print(modelo)

#predict
teste = compute(modelo,iristeste[,1:4])
teste$net.result
resultado = as.data.frame(teste$net.result)
resultado #probablidade de cada atributo ser da classe especifica
plot(modelo)

#Atribumos nomes as coluns conforme a classe
names(resultado)[1] <- 'setosa'
names(resultado)[2] <- 'versicolor'
names(resultado)[3] <- 'virginica'
resultado

#Usamos o nome das colunas para prencher uma coluna com a classe
#pegando a maior probablidade 
resultado$class = colnames(resultado[,1:3])[max.col(resultado[,1:3], ties.method = 'first')]
resultado

#Avaliamos a performance
confusao = table(resultado$class,iristeste$Species)
confusao
sum(diag(confusao) * 100 / sum(confusao))

