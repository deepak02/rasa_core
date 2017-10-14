from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import six

from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.converters import load_data
from rasa_nlu.model import Trainer

if six.PY2:
    model_name1 = 'current_py2'
else:
    model_name1 = 'current_py3'



def train_phonequery_nlu():
    print ("Let's see model ", model_name1)
    training_data = load_data('examples/phoneQuery/data/myexmp_dialog_nlu.json')
    trainer = Trainer(RasaNLUConfig("examples/phoneQuery/data/config_nlu.json"))
#    trainer = Trainer(RasaNLUConfig("examples/phoneQuery/data/config_nlu_my.json"))
    trainer.train(training_data)
    model_directory = trainer.persist('examples/phoneQuery/models/nlu/',
                                      fixed_model_name=model_name1)
    return model_directory


if __name__ == '__main__':
    logging.basicConfig(level="INFO")
    train_phonequery_nlu()
