import  nltk
from nltk.metrics import ConfusionMatrix
import BasesFrases.Base_treinamento
import BasesFrases.Base_teste
import BasesFrases.Stop_words


# baixar atualizacoes
#nltk.download()

# variaveis
baseTreinamento = BasesFrases.Base_treinamento.vet_baseTreinamento
baseTeste = BasesFrases.Base_teste.vet_baseTeste
stopWords = BasesFrases.Stop_words.stopWordsNLTK

print(baseTreinamento)
print(baseTeste)
print(stopWords)