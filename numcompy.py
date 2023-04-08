import abc
import math
import numpy as np


class Helper:

    @staticmethod
    def vector_norm(vector):
        return np.sqrt(np.sum(x**2))

    @staticmethod
    def jacobi_form(functions, vector):
        epsilon = 1e-12
        gradient_functions = []
        for function in functions:
            gradient_functions_row = []
            for index in range(len(vector)):
                epsilon_vector = np.zeros(vector.shape)
                epsilon_vector[index] = epsilon
                epsilon_added_matrix = vector.copy() + epsilon_vector
                gradient_functions_row.append(
                    (function(epsilon_added_matrix) - function(vector)) / epsilon)
            gradient_functions.append(gradient_functions_row)
        return np.array(gradient_functions)


class EquationSolverInterface(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'solve') and
                callable(subclass.solve) or
                NotImplemented)

    @abc.abstractmethod
    def solve(self):
        raise NotImplementedError


class BisectionMethodEquationSolver(EquationSolverInterface):

    def __init__(self, function, first_point, second_point, precision=1e-5, maximum_iterations=2**8):
        self.function = function
        self.negative_point = first_point
        self.positive_point = second_point
        self.precision = precision
        self.maximum_iterations = maximum_iterations
        negative_point_value = self.function(self.negative_point)
        positive_point_value = self.function(self.positive_point)
        if (negative_point_value > 0 and positive_point_value > 0) or (negative_point_value < 0 and positive_point_value < 0):
            raise Exception('invalid given points')
        if negative_point_value > 0:
            self.negative_point, self.positive_point = self.positive_point, self.negative_point

    def solve(self):
        negative_point, positive_point = self.negative_point, self.positive_point
        iteration_number = 0
        while True:
            iteration_number += 1
            next_point = (negative_point + positive_point) / 2
            next_point_value = self.function(next_point)
            if next_point_value > 0:
                positive_point = next_point
            else:
                negative_point = next_point
            error = abs(negative_point - positive_point)
            if error < self.precision:
                return next_point, error
            if iteration_number == self.maximum_iterations:
                raise Exception(
                    f'reached maximum iterations with value: {next_point} and error: {error}')


class FalsePositionMethodEquationSolver(EquationSolverInterface):

    def __init__(self, function, first_point, second_point, precision=1e-5, maximum_iterations=2**8):
        self.function = function
        self.negative_point = first_point
        self.positive_point = second_point
        self.precision = precision
        self.maximum_iterations = maximum_iterations
        self.negative_point_value = self.function(self.negative_point)
        self.positive_point_value = self.function(self.positive_point)
        if (self.negative_point_value > 0 and self.positive_point_value > 0) or (self.negative_point_value < 0 and self.positive_point_value < 0):
            raise Exception('invalid given points')
        if self.negative_point_value > 0:
            self.negative_point_value, self.positive_point_value = self.positive_point_value, self.negative_point_value
            self.negative_point, self.positive_point = self.positive_point, self.negative_point

    def solve(self):
        negative_point, positive_point = self.negative_point, self.positive_point
        negative_point_value, positive_point_value = self.negative_point_value, self.positive_point_value
        iteration_number = 0
        while True:
            iteration_number += 1
            next_point = (negative_point * positive_point_value - positive_point *
                          negative_point_value)/(positive_point_value - negative_point_value)
            next_point_value = self.function(next_point)
            if next_point_value > 0:
                positive_point = next_point
                positive_point_value = next_point_value
            else:
                negative_point = next_point
                negative_point_value = next_point_value
            error = abs(negative_point - positive_point)
            if error < self.precision:
                return next_point, error
            if iteration_number == self.maximum_iterations:
                raise Exception(
                    f'reached maximum iterations with value: {next_point} and error: {error}')


class NewtonMethodEquationSolver(EquationSolverInterface):

    def __init__(self, function, derivative_function, starting_point, precision=1e-5, maximum_iterations=2**8):
        self.function = function
        self.derivative_function = derivative_function
        self.starting_point = starting_point
        self.precision = precision
        self.maximum_iterations = maximum_iterations

    def solve(self):
        previous_point = self.starting_point
        iteration_number = 0
        while True:
            iteration_number += 1
            next_point = previous_point - \
                self.function(previous_point) / \
                self.derivative_function(previous_point)
            error = abs(next_point - previous_point)
            previous_point = next_point
            if error < self.precision:
                return next_point, error
            if iteration_number == self.maximum_iterations:
                raise Exception(
                    f'reached maximum iterations with value: {next_point} and error: {error}')


class SecantMethodEquationSolver(EquationSolverInterface):

    def __init__(self, function, first_point, second_point, precision=1e-5, maximum_iterations=2**8):
        self.function = function
        self.first_point = first_point
        self.second_point = second_point
        self.precision = precision
        self.maximum_iterations = maximum_iterations
        self.first_point_value = self.function(self.first_point)
        self.second_point_value = self.function(self.second_point)

    def solve(self):
        first_point, second_point = self.first_point, self.second_point
        first_point_value, second_point_value = self.first_point_value, self.second_point_value
        iteration_number = 0
        while True:
            iteration_number += 1
            next_point = (first_point * second_point_value - second_point *
                          first_point_value) / (second_point_value - first_point_value)
            next_point_value = self.function(next_point)
            first_point = second_point
            first_point_value = second_point_value
            second_point = next_point
            second_point_value = next_point_value
            error = abs(first_point - second_point)
            if error < self.precision:
                return next_point, error
            if iteration_number == self.maximum_iterations:
                raise Exception(
                    f'reached maximum iterations with value: {next_point} and error: {error}')


class FixedPointMethodEquationSolver(EquationSolverInterface):

    def __init__(self, function, starting_point, precision=1e-5, maximum_iterations=2**8):
        self.function = function
        self.starting_point = starting_point
        self.precision = precision
        self.maximum_iterations = maximum_iterations

    def solve(self):
        previous_point = self.starting_point
        iteration_number = 0
        while True:
            iteration_number += 1
            next_point = self.function(previous_point)
            error = abs(next_point - previous_point)
            previous_point = next_point
            if error < self.precision:
                return next_point, error
            if iteration_number == self.maximum_iterations:
                raise Exception(
                    f'reached maximum iterations with value: {next_point} and error: {error}')


class SquareMatrix:

    def __init__(self, matrix, dimension):
        self.matrix = matrix
        self.dimension = dimension

    def lower_and_upper_triangular_decomposition(self):
        upper = self.matrix.copy()
        lower = np.identity(self.dimension)
        for column in range(self.dimension):
            if upper[column][column] == 0:
                found = False
                for new_row in range(column + 1, self.dimension):
                    if upper[new_row][column] != 0:
                        found = True
                        upper[[new_row, column]] = upper[[column, new_row]]
                        break
                if not found:
                    raise Exception('singular matrix')
            for row in range(column + 1, self.dimension):
                ratio = upper[row][column] / upper[column][column]
                lower[row][column] = ratio
                upper[row] -= ratio * upper[column]
        return (lower, upper)

    def determinant(self):
        _, upper = self.lower_and_upper_triangular_decomposition()
        determinant = 1
        for index in range(self.dimension):
            determinant *= upper[index][index]
        return determinant

    def inverse(self):
        upper = self.matrix.copy()
        inverse = np.identity(self.dimension)
        for column in range(self.dimension):
            for row in range(column + 1, self.dimension):
                ratio = upper[row][column] / upper[column][column]
                upper[row] -= ratio * upper[column]
                inverse[row] -= ratio * inverse[column]
        for row in range(self.dimension - 1, -1, -1):
            for column in range(row - 1, -1, -1):
                ratio = upper[column][row] / upper[row][row]
                inverse[column] -= ratio * inverse[row]
        for index in range(self.dimension):
            inverse[index] /= upper[index][index]
        return inverse

    def delta(self, delta):
        determinant_list = []
        for index in range(self.dimension):
            delta_matrix = self.matrix.copy()
            delta_matrix[:, index] = delta
            delta_matrix = SquareMatrix(delta_matrix, self.dimension)
            determinant_list.append(delta_matrix.determinant())
        return np.array(determinant_list)


class LinearEquationsSystemSolverInterface(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'solve') and
                callable(subclass.solve) or
                NotImplemented)

    @abc.abstractmethod
    def solve(self):
        raise NotImplementedError


class GaussianMethodLinearEquationsSystemSolver(LinearEquationsSystemSolverInterface):

    def __init__(self, coefficient_matrix, constant_matrix):
        self.dimension = coefficient_matrix.dimension
        self.coefficient_matrix = coefficient_matrix.matrix
        self.constant_matrix = constant_matrix

    def solve(self):
        coefficient_matrix = self.coefficient_matrix.copy()
        constant_matrix = self.constant_matrix.copy()
        for column in range(self.dimension):
            for row in range(column + 1, self.dimension):
                ratio = coefficient_matrix[row][column] / \
                    coefficient_matrix[column][column]
                coefficient_matrix[row] -= ratio * coefficient_matrix[column]
                constant_matrix[row] -= ratio * constant_matrix[column]
        variable_matrix = np.zeros(self.dimension)
        for row in range(self.dimension - 1, -1, -1):
            variable_matrix[row] = constant_matrix[row]
            for column in range(self.dimension - 1, row, -1):
                variable_matrix[row] -= coefficient_matrix[row][column] * \
                    variable_matrix[column]
            variable_matrix[row] /= coefficient_matrix[row][row]
        return variable_matrix


class CramerMethodLinearEquationsSystemSolver(LinearEquationsSystemSolverInterface):

    def __init__(self, coefficient_matrix, constant_matrix):
        self.dimension = coefficient_matrix.dimension
        self.coefficient_matrix = coefficient_matrix.matrix
        self.coefficient_determinant = coefficient_matrix.determinant()
        self.constant_matrix = constant_matrix

    def solve(self):
        constant_matrix = self.constant_matrix.copy()
        variable_matrix = np.zeros(self.dimension)
        for index in range(self.dimension):
            coefficient_matrix = self.coefficient_matrix.copy().T
            coefficient_matrix[index] = constant_matrix
            variable_matrix[index] = SquareMatrix(
                coefficient_matrix, self.dimension).determinant() / self.coefficient_determinant
        return variable_matrix


class PrincipalElementMethodLinearEquationsSystemSolver(LinearEquationsSystemSolverInterface):

    def __init__(self, coefficient_matrix, constant_matrix):
        self.dimension = coefficient_matrix.dimension
        self.coefficient_matrix = coefficient_matrix.matrix
        self.constant_matrix = constant_matrix

    def solve(self):
        dimension = self.dimension
        coefficient_matrix = self.coefficient_matrix.copy()
        constant_matrix = self.constant_matrix
        principal_elements = []
        removed_elements = np.zeros(dimension)
        removed_rows = np.zeros(dimension)
        removed_columns = np.zeros(dimension)

        for index in range(dimension):
            argument_max = coefficient_matrix.argmax()
            row_number = argument_max % dimension
            column_number = argument_max // dimension
            actual_row = 0
            row_copy = row_number
            while removed_rows[actual_row] == 1:
                actual_row += 1
            while row_copy > 0:
                while removed_rows[actual_row] == 1:
                    actual_row += 1
                if removed_rows[actual_row] == 0:
                    row_copy -= 1
                actual_row += 1
            removed_rows[actual_row] = 1

            actual_column = 0
            column_copy = column_number
            while removed_columns[actual_column] == 1:
                actual_column += 1
            while column_copy > 0:
                while removed_columns[actual_column] == 1:
                    actual_column += 1
                if removed_columns[actual_column] == 0:
                    column_copy -= 1
                actual_column += 1
            removed_columns[actual_column] = 1

            principal_elements.append(
                (coefficient_matrix[column_number], actual_row, actual_column))
            ratio = coefficient_matrix[column_number][row_number]
            removed_elements[actual_column] = constant_matrix[column_number]
            for column_index in range(dimension):
                if column_index == column_number:
                    continue
                column_ratio = coefficient_matrix[column_index][row_number] / ratio
                coefficient_matrix[column_index] -= column_ratio * \
                    coefficient_matrix[column_number]
                constant_matrix[column_index] -= column_ratio * \
                    constant_matrix[column_number]
            coefficient_matrix = np.delete(coefficient_matrix, row_number, 1)
            coefficient_matrix = np.delete(
                coefficient_matrix, column_number, 0)
            constant_matrix = np.delete(constant_matrix, column_number, 0)
            dimension -= 1

        dimension = self.dimension
        variable_matrix = np.zeros(dimension)
        row_numbers = []
        for index in range(dimension - 1, -1, -1):
            column, row_number, column_number = principal_elements[index]
            row_numbers.append(row_number)
            row_numbers.sort()
            variable_matrix[row_number] = removed_elements[column_number]
            final_row_number = 0
            for i, listed_row_number in enumerate(row_numbers):
                if listed_row_number == row_number:
                    final_row_number = i
                else:
                    variable_matrix[row_number] -= variable_matrix[listed_row_number] * column[i]
            variable_matrix[row_number] /= column[final_row_number]
        return variable_matrix


class NonLinearEquationsSystemSolverInterface(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'solve') and
                callable(subclass.solve) or
                NotImplemented)

    @abc.abstractmethod
    def solve(self):
        raise NotImplementedError


class NewtonMethodNonLinearEquationsSystemSolver(NonLinearEquationsSystemSolverInterface):

    def __init__(self, functions, starting_variables, precision=1e-5, maximum_iterations=2**8):
        self.dimension = len(functions)
        self.functions = functions
        self.starting_variables = starting_variables
        self.precision = precision
        self.maximum_iterations = maximum_iterations

    def solve(self):
        previous_vector = self.starting_variables
        iteration_number = 0
        while True:
            iteration_number += 1
            delta = np.array([function(previous_vector)
                             for function in self.functions])
            jacobi_form = SquareMatrix(Helper.jacobi_form(
                self.functions, previous_vector), self.dimension)
            next_vector = previous_vector - \
                jacobi_form.delta(delta) / jacobi_form.determinant()
            error = np.abs(next_vector - previous_vector)
            previous_vector = next_vector
            if Helper.vector_norm(error) < self.precision:
                return next_vector, error
            if iteration_number == self.maximum_iterations:
                raise Exception(
                    f'reached maximum iterations with value: {next_vector} and error: {error}')


class InterpolatorInterface(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'interpolate') and
                callable(subclass.interpolate) or
                NotImplemented)

    @abc.abstractmethod
    def interpolate(self):
        raise NotImplementedError


class NewtonMethodInterpolator(InterpolatorInterface):

    def __init__(self, points):
        points_number = len(points)
        x = np.array([points[i][0] for i in range(points_number)])
        y = np.array([points[i][1] for i in range(points_number)])
        coefficients = np.copy(y)
        for i in range(1, points_number):
            coefficients[i:points_number] = (
                coefficients[i:points_number] - coefficients[i - 1]) / (x[i:points_number] - x[i - 1])

        def interpolated_function(input):
            polynomial_degree = points_number - 1
            result = coefficients[polynomial_degree]
            for k in range(1, polynomial_degree + 1):
                result = coefficients[polynomial_degree - k] + \
                    (input - x[polynomial_degree - k]) * result
            return result

        self.interpolated_function = interpolated_function

    def interpolate(self, input):
        return self.interpolated_function(input)


class LagrangeMethodInterpolator(InterpolatorInterface):

    def __init__(self, points):
        points_number = len(points)

        def interpolated_function(input):
            result = 0
            for i in range(points_number):
                tmp = points[i][1]
                for j in range(points_number):
                    if i != j:
                        tmp *= (input - points[j][0]) / \
                            (points[i][0] - points[j][0])
                result += tmp
            return result

        self.interpolated_function = interpolated_function

    def interpolate(self, input):
        return self.interpolated_function(input)


class NevilleMethodInterpolator(InterpolatorInterface):

    def __init__(self, points):
        points_number = len(points)
        x = np.array([points[i][0] for i in range(points_number)])
        y = np.array([points[i][1] for i in range(points_number)])

        def interpolated_function(input):
            result = np.zeros((points_number, points_number))
            result[:, 0] = y
            for i in range(1, points_number):
                for j in range(i, points_number):
                    result[j, i] = ((x[j] - input) * result[j - 1, i - 1] -
                                    (x[j - i] - input) * result[j, i - 1]) / (x[j] - x[j - i])
            return result[-1, -1]

        self.interpolated_function = interpolated_function

    def interpolate(self, input):
        return self.interpolated_function(input)


class IntegratorInterface(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'integrate') and
                callable(subclass.integrate) or
                NotImplemented)

    @abc.abstractmethod
    def integrate(self):
        raise NotImplementedError


class TrapezoidalMethodIntegrator(IntegratorInterface):

    def __init__(self, function):
        self.function = function

    def integrate(self, points, degree=10):
        points = (min(points), max(points))
        h = (points[1] - points[0]) / degree
        result = (self.function(points[0]) + self.function(points[1])) / 2
        result += sum(self.function(points[0] + i * h)
                      for i in range(1, degree))
        result *= h
        return result


class SimpsonMethodIntegrator(IntegratorInterface):

    def __init__(self, function):
        self.function = function

    def integrate(self, points, degree=10):
        points = (min(points), max(points))
        h = (points[1] - points[0]) / degree
        result = self.function(points[0]) + self.function(points[1])
        result += sum((2 if i % 2 == 0 else 4) *
                      self.function(points[0] + i * h) for i in range(1, degree))
        result *= h / 3
        return result


class DifferentialEquationNumericSolverInterface(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'solve') and
                callable(subclass.solve) or
                NotImplemented)

    @abc.abstractmethod
    def solve(self):
        raise NotImplementedError


class RungeKutta:

    def __call__(self, degree):
        if degree == 1:
            return FirstDegreeRungeKuttaDifferentialEquationSolver
        elif degree == 2:
            return SecondDegreeRungeKuttaDifferentialEquationSolver
        elif degree == 3:
            return ThirdDegreeRungeKuttaDifferentialEquationSolver
        elif degree == 4:
            return FourthDegreeRungeKuttaDifferentialEquationSolver
        else:
            raise ValueError('degree should be an integer between 1 and 4!')


class FirstDegreeRungeKuttaDifferentialEquationSolver(DifferentialEquationNumericSolverInterface):

    def __init__(self, function):
        self.function = function

    def solve(self, points, n, initial_value):
        points = (min(points), max(points))
        h = (points[1] - points[0]) / n
        y = initial_value
        answers = []

        for i in range(n):
            x = points[0] + i * h
            k1 = h * self.function(x, y)
            y += k1
            answers.append((x + h, y))
        
        return answers


class SecondDegreeRungeKuttaDifferentialEquationSolver(DifferentialEquationNumericSolverInterface):

    def __init__(self, function):
        self.function = function

    def solve(self, points, n, initial_value):
        points = (min(points), max(points))
        h = (points[1] - points[0]) / n
        y = initial_value
        answers = []

        for i in range(n):
            x = points[0] + i * h
            k1 = h * self.function(x, y)
            k2 = h * self.function(x + h, y + k1)
            y += (k1 + k2) / 2
            answers.append((x + h, y))
        
        return answers


class ThirdDegreeRungeKuttaDifferentialEquationSolver(DifferentialEquationNumericSolverInterface):

    def __init__(self, function):
        self.function = function

    def solve(self, points, n, initial_value):
        points = (min(points), max(points))
        h = (points[1] - points[0]) / n
        y = initial_value
        answers = []

        for i in range(n):
            x = points[0] + i * h
            k1 = h * self.function(x, y)
            k2 = h * self.function(x + h / 2, y + k1 / 2)
            k3 = h * self.function(x + h, y - k1 + 2 * k2)
            y += (k1 + 4 * k2 + k3) / 6
            answers.append((x + h, y))
        
        return answers


class FourthDegreeRungeKuttaDifferentialEquationSolver(DifferentialEquationNumericSolverInterface):

    def __init__(self, function):
        self.function = function

    def solve(self, points, n, initial_value):
        points = (min(points), max(points))
        h = (points[1] - points[0]) / n
        y = initial_value
        answers = []

        for i in range(n):
            x = points[0] + i * h
            k1 = h * self.function(x, y)
            k2 = h * self.function(x + h / 2, y + k1 / 2)
            k3 = h * self.function(x + h / 2, y + k2 / 2)
            k4 = h * self.function(x + h, y + k3)
            y += (k1 + 2 * k2 + 2 * k3 + k4) / 6
            answers.append((x + h, y))
        
        return answers
