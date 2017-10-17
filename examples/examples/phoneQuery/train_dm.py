from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from examples.phonequery_example import phoneQueryPolicy
from rasa_core.agent import Agent
from rasa_core.policies.memoization import MemoizationPolicy


def train_phoneQuery_dm():
    training_data_file = 'examples/phoneQuery/data/phonequery.md'
    model_path = 'examples/phoneQuery/models/policy/current'

    agent = Agent("examples/phonequery_domain.yml",
                  policies=[MemoizationPolicy(), phoneQueryPolicy()])

    agent.train(
            training_data_file,
            max_history=3,
            epochs=100,
            batch_size=50,
            augmentation_factor=50,
            validation_split=0.2
    )

    agent.persist(model_path)


if __name__ == '__main__':
    logging.basicConfig(level="DEBUG")
    train_phoneQuery_dm()
