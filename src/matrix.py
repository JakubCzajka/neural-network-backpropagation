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



def dot_product(vector_a: List[float], vector_b: List[float]):
    return sum([a * b for a, b in zip(vector_a, vector_b)])


class Matrix():
    def __init__(self, columns: int, rows: int, func):
        self.matrix = []
        self.columns = columns
        self.rows = rows

        for col in range(columns):
            self.matrix.append([])

            for _ in range(rows):
                self.matrix[col].append(func())

    def get(self, column: int, row: int):
        return self.matrix[column][row]

    
    def get_row(self, row: int, as_list = True):
        if as_list:
            return [self.matrix[col][row] for col in range(self.columns)]
        else:
            result = Matrix(self.columns, 1, lambda : 0)
            for col in range(self.columns):
                result.set(col, 0, self.get(col, row))
            return result

    
    def get_column(self, column: int, as_list = True):
        if as_list:
            return [self.matrix[column][row] for row in range(self.rows)]
        else:
            result = Matrix(self.rows, 1, lambda : 0)
            for row in range(self.rows):
                result.set(row, 0, self.get(column, row))
            return result

    
    def set(self, column: int, row: int, new_value: float):
        self.matrix[column][row] = new_value

    
    def transpose(self):
        result = Matrix(self.rows, self.columns, lambda : 0)

        for col in range(self.columns):
            for row in range(self.rows):
                result.set(row, col, self.matrix[col][row])
        
        return result

    
    def multiply(self, other_or_value):
        if isinstance(other_or_value, float) or isinstance(other_or_value, int):
            result = Matrix(self.columns, self.rows, lambda : 0)
            value = other_or_value
            for col in range(self.columns):
                for row in range(self.rows):
                    result.matrix[col][row] = self.matrix[col][row] * value

            return result
            


        elif isinstance(other_or_value, Matrix):
            other = other_or_value

            assert self.columns == other.rows

            result = Matrix(other.columns, self.rows, lambda : 0)

            for col in range(other.columns):
                for row in range(self.rows):
                    result.set(col, row, dot_product(self.get_row(row), other.get_column(col)))

            return result

    
    def multiply_element_wise(self, other):
        assert self.columns == other.columns
        assert self.rows == other.rows

        result = Matrix(self.columns, self.rows, lambda : 0)

        for col in range(other.columns):
            for row in range(self.rows):
                result.set(col, row, self.get(col, row) * other.get(col, row))

        return result

    
    def add(self, other: 'Matrix'):
        assert self.columns == other.columns
        assert self.rows == other.rows

        result = Matrix(self.columns, self.rows, lambda : 0)   

        for col in range(self.columns):
            for row in range(self.rows):
                result.set(col, row, self.get(col, row) + other.get(col, row))

        return result

    
    def apply(self, func):
        result = Matrix(self.columns, self.rows, lambda : 0)   

        for col in range(self.columns):
            for row in range(self.rows):
                result.set(col, row, func(self.get(col, row)))

        return result

    
    def print(self):
        for row in range(self.rows):
            print(self.get_row(row))

    
    @staticmethod
    def read_dataset(dataset: pd.DataFrame):
        rows = dataset.index.size
        columns = dataset.columns.size

        converter = Converter(dataset)

        values = Matrix(columns-1, rows, lambda : 0)
        classes = Matrix(converter.size, rows, lambda : 0)
        i=0
        for _, row_values in dataset.iterrows():

            row = row_values.to_dict()

            class_name = row.pop('class')

            for col, value in enumerate(row.values()):
                values.set(col, i, value)

            for col, value in enumerate(converter.encode(class_name)):
                classes.set(col, i, value)
            i+=1

        return values, classes