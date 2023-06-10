# Fuzzy comparing by sentence similarity
# author: Vladislav Janvarev

import os

from vacore import VACore

modname = os.path.basename(__file__)[:-3] # calculating modname

model = None
# функция на старте
def start(core:VACore):
    manifest = {
        "name": "Fuzzy compare by sentence similarity",
        "version": "1.0",
        "require_online": False,

        "fuzzy_processor": {
            "ai_sentence": (init,predict) # первая функция инициализации, вторая - обработка
        }
    }
    return manifest

def init(core:VACore):
    global model
    import sentence_transformers

    model = sentence_transformers.SentenceTransformer('inkoziev/sbert_synonymy')

def predict(core:VACore, command:str, context:dict, allow_rest_phrase:bool = True): # на входе -
            # команда; текущий контекст в формате Ирины; разрешен ли остаток фразы, или это фраза целиком
    import sentence_transformers

    bestres = 0
    bestret = None

    if allow_rest_phrase: # разрешены остатки фраз? сравниваем только с началом
        #cmdsub = command[0:len(key)]
        #rest_phrase = command[(len(key)+1):]
        cmdsub = command
        rest_phrase = ""
    else:
        cmdsub = command
        rest_phrase = ""

    embeddings_user = model.encode([cmdsub])[0]

    for keyall in context.keys():
        keys = keyall.split("|")
        for key in keys:
            # print("Subcmd: ",cmdsub,key)

            # для всех ключей вычисляем схожесть
            emb_cur_key = model.encode([key])[0]
            # можно конечно, оптимизировать, и не запускать encode по нескольку раз, а только 1 для всего списка

            res = sentence_transformers.util.cos_sim(a=embeddings_user, b=emb_cur_key).item()

            if res > bestres:
                bestres = res
                bestret = (keyall,float(res),rest_phrase)


    return bestret


