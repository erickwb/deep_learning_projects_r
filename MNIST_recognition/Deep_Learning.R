'''
Name: Erick Correia Silva
classification of MNIST data
'''

library(h2o)

#importação o arquivo para visualização
#uso interno
options(warn=-1)
digitos <- read.csv(gzfile("test.csv.gz"), header=F)
dim(digitos)
head(digitos)

#visualizando alguns digitos
#criando uma matriz 28x28
dig1 = t(matrix(unlist(digitos[20,-785]), nrow = 28, byrow = F))
dig1 = t(apply(dig, 2, rev))
dig1
digitos[20,785]

dig2 = t(matrix(unlist(digitos[2,-785]), nrow = 28, byrow = F))
dig2 = t(apply(dig2, 2, rev))

dig3 = t(matrix(unlist(digitos[4,-785]), nrow = 28, byrow = F))
dig3 = t(apply(dig3, 2, rev))

dig4 = t(matrix(unlist(digitos[5,-785]), nrow = 28, byrow = F))
dig4 = t(apply(dig4, 2, rev))

#visualizando as imagens
#Executar com ctrl + shift + enter
image(dig1, col = grey.colors(255))
image(dig2,col=grey.colors(255))
image(dig3,col=grey.colors(255))
image(dig4,col=grey.colors(255))

#inicilizando o cluster h2o
h2o.init()
#importando os dados com a funçao importfile do h2o
treino <- h2o.importFile('train.csv.gz')
teste <- h2o.importFile('test.csv.gz')
dim(treino)
head(treino)
colnames(treino)
#transforma a classe em fator
treino[,785] <- as.factor(treino[,785])
teste[,785] <- as.factor(teste[,785])

#Modelo
#obs: o modelo ja faz uma cross validation
modelo <- h2o.deeplearning(x = colnames(treino[,1:784]),  y = "C785",  training_frame = treino,  validation_frame = teste,  distribution = "AUTO",  activation = "RectifierWithDropout",  hidden = c(64,64,64),  sparse = TRUE, epochs = 20)
plot(modelo)

#Performance do modelo
h2o.performance(modelo)

