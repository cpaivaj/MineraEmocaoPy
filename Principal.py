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

# extracao das palavras de uma frase que foram passada por parametro
def extratorPalavras(documento):
    doc = set(documento)
    caracteristicas = {}
    for palavras in palavrasUnicasTreinamento:
        # para percorrer as palavras de uma frase
        caracteristicas['%s' % palavras] = (palavras in doc)
    return caracteristicas

# passa como parametro radicais de palavras
# mostra se existe a palavra na frase ou nao, com tru ou false
# extrai as que estao sendo passadas por paramento
caracteristicasFrase = extratorPalavras(['am', 'nov', 'dia'])

# criacao das bases completas ja modificadas
# passa como parametros a funcao que extrai as palavras e a var frasescomstemming, pq a base ja esta processada
# essa funcao apply_features, ja faz o processamento necessario para toda a base
baseCompletaTreinamento = nltk.classify.apply_features(extratorPalavras, frasesComStemmingTreinamento)
baseCompletaTeste = nltk.classify.apply_features(extratorPalavras, frasesComStemmingTeste)

print(baseCompletaTreinamento)
print(baseCompletaTeste)
