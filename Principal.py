import  nltk
from nltk.metrics import ConfusionMatrix
import BasesFrases.Base_treinamento
import BasesFrases.Base_teste
import BasesFrases.Stop_words
import Stemmer

# baixar atualizacoes
#nltk.download()

# variaveis
baseTreinamento = BasesFrases.Base_treinamento.vet_baseTreinamento
baseTeste = BasesFrases.Base_teste.vet_baseTeste
stopWords = BasesFrases.Stop_words.stopWordsNLTK

# aplicando stemming
frasesComStemmingTreinamento = Stemmer.aplicaStemmer(baseTreinamento)
frasesComStemmingTeste = Stemmer.aplicaStemmer(baseTeste)

print(frasesComStemmingTreinamento)
print(frasesComStemmingTeste)