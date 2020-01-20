__author__ = 'diego'

class OieExample (object):

    def __init__(self, arg1, arg2, features, trigger, relation=''):
        self.features = features
        self.arg1 = arg1
        self.arg2 = arg2
        self.relation = relation
        self.trigger = trigger

    def setFeatures(self, features):
        self.features = features
    
    def __str__(self):
        return str({'arg1': self.arg1,'arg2': self.arg2,'relation': self.relation,'trigger': self.trigger,'features': self.features,})