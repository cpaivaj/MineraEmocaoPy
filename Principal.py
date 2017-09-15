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

# busca cada uma das palavras, apos quebrar todas em radicais
palavrasTreinamento = Stemmer.buscaPalavras(frasesComStemmingTreinamento)
palavrasTeste = Stemmer.buscaPalavras(frasesComStemmingTeste)

# quantidade de vezes que uma palavra se repete
frequenciaTreinamento = Stemmer.buscaFrequencia(palavrasTreinamento)
frequenciaTeste = Stemmer.buscaFrequencia(palavrasTeste)

# palavras que nao se repetem
palavrasUnicasTreinamento = Stemmer.buscaPalavrasUnicas(frequenciaTreinamento)
palavrasUnicasTeste = Stemmer.buscaPalavrasUnicas(frequenciaTeste)

# passa como parametro radicais de palavras
# mostra se existe a palavra na frase ou nao, com tru ou false
# extrai as que estao sendo passadas por paramento
caracteristicasFrase = Stemmer.extratorPalavras(['am', 'nov', 'dia'], palavrasUnicasTreinamento)

print(caracteristicasFrase)
