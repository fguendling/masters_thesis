# Original Script: evaltopic.py
# Verwendet (vertrauliche) Daten aus einer Datenbank ist nur bei Kenntnis der Zugangsdaten ausführbar.

def individual_EvalWordVector():
    
    connectdb() # Es wird hier die lokale Instanz verwendet (MacBook)
    
    onts = OntologyDefinition.GetAll("DGDTST")

    for ont in onts:
        ontwv_qry = OntologyWordVector.objects(ProjectKey="DGDTST", OntologyKey = ont.OntologyKey)
        if ontwv_qry.count() == 0:
            ont.WordVector = None
        else:
            ont.WordVector = ontwv_qry[0] #avg vector
    
    _log("calcing similarity")
    for ont in onts:
        if ont.OntologyKey != "GEN":
            if ont.WordVector != None:
                #print(vec)
                #print(ont.WordVector)      
                #print(ont.OntologyKey, ": ", 0.0 + ont.WordVector.Similarity(vec))
                #_AggrTopicValuesDic[ont.OntologyKey] = [ont.OntologyKey, 0.0 + ont.WordVector.Similarity(vec)]
                #print(ont.OntologyKey)                
                for o in onts:
                    if ont.OntologyKey != "GEN":
                        if o.WordVector != None:
                            print(ont.OntologyKey, "&", o.OntologyKey, ": ", ont.WordVector.Similarity(o.WordVector))
            else:
                pass

    #wvlist = [wvrec for wvrec in _AggrTopicValuesDic.values() if wvrec[1] > 0]
    #_finalontlist, _BestGuess, _finalvote = _GetEvaluationResults(_eval, wvlist, "WV")
    #totaleval[_eval.ChapterId] = wvlist

individual_EvalWordVector()    