from typing import List, Tuple
import random
import pandas as pd
from converter import Converter



def dot_product(vector_a: List[float], vector_b: List[float]):
    return sum([a * b for a, b in zip(vector_a, vector_b)])


class Matrix():
    def __init__(self, columns: int, rows: int, func: callable):
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


def split_dataset(dataset: Tuple[Matrix, Matrix], rate: float):

    def create_matrix_from_rows(rows):
        result = Matrix(len(rows[0]), len(rows), lambda : 0)

        for row_no, row in enumerate(rows):
            for col_no, value in enumerate(row):
                result.set(col_no, row_no, value)

        return result

    x = dataset[0]
    y = dataset[1]

    x_1 = []
    y_1 = []

    x_2 = []
    y_2 = []

    for row in range(x.rows):
        if random.uniform(0.0, 1.0) < rate:
            x_1.append(x.get_row(row))
            y_1.append(y.get_row(row))
        else:
            x_2.append(x.get_row(row))
            y_2.append(y.get_row(row))

    x_train = create_matrix_from_rows(x_1)
    y_train = create_matrix_from_rows(y_1)


    x_test = create_matrix_from_rows(x_2)
    y_test = create_matrix_from_rows(y_2)

    return (x_train, y_train), (x_test, y_test)
