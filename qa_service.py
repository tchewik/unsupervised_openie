
from bentoml import BentoService, api, env, artifacts
from bentoml.artifact import TextFileArtifact
from bentoml.handlers import JsonHandler

import flair
from flair.data import Sentence
from flair.models import SequenceTagger

from py2neo import Graph

import json

import pandas as pd

@env(pip_dependencies=['flair', 'torch', 'pandas', 'py2neo', 'numpy'])
@artifacts([TextFileArtifact('relations')])
class QAService(BentoService):
    
    def get_entities(self, the_question, model):
        the_sentenced_question = Sentence(the_question)
        model.predict(the_sentenced_question)
        spans = [span for span in the_sentenced_question.get_spans('ner') if span.tag == "PER" or span.tag == "MISC" or span.tag == 'LOC']
        entities = [" ".join([tok.text for tok in span.tokens]) for span in spans]
        return entities
    
    def query(self, graph, subject_name = None, relation_type = None, object_name = None):
        """
        At least two arguments should not be None for adequate amount of results
        """
        query = "MATCH (s:object)-[r:predicate]->(o:object) [WHERE_CLAUSE]RETURN s.name as subject, r.type as relation, o.name as object"

        clause = "WHERE "
        used = False
        if subject_name is not None:
            clause += f"s.name = '{subject_name}' "
            used = True
        if relation_type is not None:
            clause_item = f"r.type = '{relation_type}' "
            if used:
                clause_item = "AND " + clause_item

            clause += clause_item
            used = True
        if object_name is not None:
            clause_item = f"o.name = '{object_name}' "
            if used:
                clause_item = "AND " + clause_item

            clause += clause_item
            used = True

        if used:
            query = query.replace("[WHERE_CLAUSE]", clause)
        else:
            query = query.replace("[WHERE_CLAUSE]", '')

        return graph.run(query).to_data_frame()
    
    def execute_question(self, question, ner_model, relations, graph):
        ##--stage 1: detect entities--##
        entities = self.get_entities(question, model=ner_model)

        true_relation_types = []
        ##--stage 2: detect relations--##
        for candidate_relation_type, candidate_relations_set in relations.items():
            if any(relation in question for relation in candidate_relations_set):
                true_relation_types.append(candidate_relation_type)

        ##--stage 3: executing queries--##
        dfs = []
        for entity in entities:
            for rel_type in true_relation_types:
                dfs.append(self.query(graph, subject_name=entity, relation_type=rel_type))
                dfs.append(self.query(graph, object_name=entity, relation_type=rel_type))

        return pd.concat(dfs, axis=0) if len(dfs) > 0 else []

    @api(JsonHandler)
    def predict(self, input_dict):
        
        credentials = {
            'host': '####',
            'username': '####',
            'password': '####'
        }
        
        ner = SequenceTagger.load('ner')
        relations = json.loads(self.artifacts.relations)
        graph = Graph(credentials['host'], auth=(credentials['username'], credentials['password']))
        question = input_dict['question']
        
        dataframe = self.execute_question(question, ner, relations, graph)
        return dataframe.to_json(orient='records') if len(dataframe) != 0 else []
        
        
