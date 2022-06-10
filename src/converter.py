from typing import List
import pandas as pd


class Converter():
    def __init__(self, dataset: pd.DataFrame):
        column_names = list(dataset['class'].unique())

        self.class_index = {}
        self.index_class = {}

        self.size = len(column_names)

        for id, name in enumerate(column_names):
            self.class_index[name] = id
            self.index_class[id] = name


    def encode(self, class_name: str):
        result = []

        one_position = self.class_index.get(class_name)

        for i in range(self.size):
            if i != one_position:
                result.append(0)
            else:
                result.append(1)

        return result

    
    def decode(self, vector: List):
        max_value = vector[0]
        max_index = 0

        for i in range(1, len(vector)):

            if vector[i] > max_value:
                max_value = vector[i]
                max_index = i

        return self.index_class.get(max_index)
