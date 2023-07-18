"""Модуль базовых алгоритмов линейной алгебры.
Задание состоит в том, чтобы имплементировать класс Matrix
(следует воспользоваться кодом из задания seminar06_1), учтя рекомендации pylint.
Для проверки кода следует использовать команду pylint matrix.py.
Pylint должен показывать 10 баллов.
Рекомендуемая версия pylint - 2.15.5
Кроме того, следует добавить поддержку исключений в отмеченных местах.
Для проверки корректности алгоритмов следует сравнить результаты с соответствующими функциями numpy.
"""
import random
import copy
import numpy as np

class Matrix:
    """Класс работы с матрицами"""
    def __init__(self, nrows, ncols, init="zeros"):
        """Конструктор класса Matrix.
        Создаёт матрицу резмера nrows x ncols и инициализирует её методом init.
        nrows - количество строк матрицы
        ncols - количество столбцов матрицы
        init - метод инициализации элементов матрицы:
            "zeros" - инициализация нулями
            "ones" - инициализация единицами
            "random" - случайная инициализация
            "eye" - матрица с единицами на главной диагонали
        """
        if nrows < 0 or ncols < 0:
            raise ValueError("nrows and ncols should be non negative")
        if init not in ["zeros", "ones", "eye", "random"]:
            raise ValueError("matrix cant be initialized that way")
        self.nrows = nrows
        self.ncols = ncols
        self.data = [[]] # Это должен быть список списков
        self.data_init(init)
    def data_init(self, init):
        """инициализация данных"""
        tmp = 0.
        if init in ("zeros","ones","eye"):
            if init == "ones":
                tmp = 1.
            for i in range(self.nrows):
                if i != self.nrows-1:
                    self.data.append([])
                for j in range(self.ncols):
                    self.data[i].append(tmp)
                    if init == "eye" and i == j:
                        self.data[i][j] = 1.
        if init == "random":
            for i in range(self.nrows):
                if i != self.nrows-1:
                    self.data.append([])
                for j in range(self.ncols):
                    self.data[i].append(random.uniform(-100.,100.))
    @staticmethod
    def from_dict(data):
        "Десериализация матрицы из словаря"
        ncols = data["ncols"]
        nrows = data["nrows"]
        items = data["data"]
        assert len(items) == ncols*nrows
        result = Matrix(nrows, ncols)
        for row in range(nrows):
            for col in range(ncols):
                result[(row, col)] = items[ncols*row + col]
        return result
    @staticmethod
    def to_dict(matr):
        "Сериализация матрицы в словарь"
        assert isinstance(matr, Matrix)
        nrows, ncols = matr.shape()
        data = []
        for row in range(nrows):
            for col in range(ncols):
                data.append(matr[(row, col)])
        return {"nrows": nrows, "ncols": ncols, "data": data}
    def __str__(self):
        string = "[ "
        for i in self.data:
            string+="[ "
            for j in i:
                string+=str(j)
                string+=" "
            string+="] "
        string+="]"
        return string
    def __repr__(self):
        return str(self)+": Matrix"
    def shape(self):
        "Вернуть кортеж размера матрицы (nrows, ncols)"
        return (self.nrows, self.ncols)
    def __getitem__(self, index):
        """Получить элемент матрицы по индексу index
        index - список или кортеж, содержащий два элемента
        """
        if not isinstance(index, list) and not isinstance(index, tuple):
            raise ValueError("index not list or tuple")
        if len(index) != 2:
            raise ValueError("not 2 elements in index")
        if index[0] < 0 or index[0] > self.nrows-1 or index[1] < 0 or index[1] > self.ncols-1:
            raise IndexError("wrong index")
        row, col = index
        return self.data[row][col]
    def __setitem__(self, index, value):
        """Задать элемент матрицы по индексу index
        index - список или кортеж, содержащий два элемента
        value - Устанавливаемое значение
        """
        if not isinstance(index, list) and not isinstance(index, tuple):
            raise ValueError("index not list or tuple")
        if len(index) != 2:
            raise ValueError("not 2 elements in index")
        if index[0] < 0 or index[0] > self.nrows-1 or index[1] < 0 or index[1] > self.ncols-1:
            raise IndexError("wrong index")
        row, col = index
        self.data[row][col] = value
    def __sub__(self, rhs):
        "Вычесть матрицу rhs и вернуть результат"
        if self.shape() != rhs.shape():
            raise ValueError("different shapes")
        result = []
        for i in range(self.nrows):
            for j in range(self.ncols):
                result.append(self.data[i][j] - rhs[i, j])
        return self.from_dict({"data": result, "ncols": self.ncols, "nrows": self.nrows})
    def __add__(self, rhs):
        "Сложить с матрицей rhs и вернуть результат"
        if self.shape() != rhs.shape():
            raise ValueError("different shapes")
        result = []
        for i in range(self.nrows):
            for j in range(self.ncols):
                result.append(self.data[i][j] + rhs[i, j])
        return self.from_dict({"data": result, "ncols": self.ncols, "nrows": self.nrows})
    def __mul__(self, rhs):
        "Умножить на матрицу rhs и вернуть результат"
        if self.ncols != rhs.shape()[0]:
            raise ValueError("different shapes")
        result = []
        for i in range(self.nrows):
            for j in range(rhs.shape()[1]):
                tmp = 0.
                for k in range(self.ncols):
                    tmp+=self.data[i][k]*rhs[k, j]
                result.append(tmp)
        return self.from_dict({"data": result, "ncols": self.nrows, "nrows": rhs.shape()[1]})
    def __pow__(self, power):
        "Возвести все элементы в степень power и вернуть результат"
        result = []
        for i in range(self.nrows):
            for j in range(self.ncols):
                result.append(self.data[i][j]**power)
        return self.from_dict({"data": result, "ncols": self.ncols, "nrows": self.nrows})
    def sum(self):
        "Вернуть сумму всех элементов матрицы"
        summa = 0.
        for i in self.data:
            for j in i:
                summa+=j
        return summa
    def det(self):
        "Вычислить определитель матрицы"
        if self.nrows != self.ncols:
            raise ArithmeticError("non quadratic matrix")
        mat = copy.deepcopy(self.data)
        det = 1.
        for i in range(self.nrows):
            if abs(mat[i][i]) < 1e-13:
                for j in range(i+1, self.nrows):
                    if mat[j][i] > 0:
                        for k in range(self.ncols):
                            tmp = mat[i][k]
                            mat[i][k] = mat[j][k]
                            mat[j][k] = tmp
                        det*=-1.
                        break
            if abs(mat[i][i]) < 1e-13:
                break
            for j in range(i+1, self.nrows):
                tmp = -mat[j][i]
                for k in range(self.ncols):
                    mat[j][k]+=tmp*mat[i][k]/mat[i][i]
        for i in range(self.nrows):
            det*=mat[i][i]
        return det
    def transpose(self):
        "Транспонировать матрицу и вернуть результат"
        result = []
        for i in range(self.ncols):
            for j in range(self.nrows):
                result.append(self.data[j][i])
        return self.from_dict({"data": result, "ncols": self.nrows, "nrows": self.ncols})
    def straight_gauss(self, mat, eye):
        """прямой ход метода гаусса"""
        for i in range(self.nrows):
            if abs(mat[i][i]) < 1e-13:
                for j in range(i+1, self.nrows):
                    if mat[j][i] > 0:
                        for k in range(self.ncols):
                            tmp1 = mat[i][k]
                            mat[i][k] = mat[j][k]
                            mat[j][k] = tmp1
                            tmp2 = eye[i*self.nrows + k]
                            eye[i*self.nrows + k] = eye[j*self.nrows + k]
                            eye[j*self.nrows + k] = tmp2
                        break
            tmp = 1/mat[i][i]
            for k in range(self.ncols):
                mat[i][k]*=tmp
                eye[i*self.nrows + k]*=tmp
            for j in range(i+1, self.nrows):
                tmp = -mat[j][i]
                for k in range(self.ncols):
                    mat[j][k]+=tmp*mat[i][k]
                    eye[j*self.nrows + k]+=tmp*eye[i*self.nrows + k]
    def inv(self):
        "Вычислить обратную матрицу и вернуть результат"
        if self.nrows != self.ncols:
            raise ArithmeticError("non quadratic matrix")
        if abs(self.det()) < 1e-13:
            raise ArithmeticError("det equals zero")
        mat = copy.deepcopy(self.data)
        eye = []
        for i in range(self.nrows):
            for j in range(self.ncols):
                eye.append(0.)
                if i == j:
                    eye[i*self.nrows + j] = 1.
        self.straight_gauss(mat,eye)
        for i in range(self.nrows-1, 0, -1):
            for j in range(i-1, -1, -1):
                tmp = -mat[j][i]
                for k in range(self.ncols):
                    mat[j][k]+=tmp*mat[i][k]
                    eye[j*self.nrows + k]+=tmp*eye[i*self.nrows + k]
        return self.from_dict({"data": eye, "ncols": self.ncols, "nrows": self.nrows})
    def tonumpy(self):
        "Приведение к массиву numpy"
        return np.array(self.data)
def test():
    """тесты"""
    mat = Matrix(5, 5, "ones")
    npmat = np.ones((5, 5))
    assert abs(mat.det()-np.linalg.det(npmat)) < 1e-13
    mat = mat - Matrix(5, 5, "eye")
    npmat = npmat - np.eye(5, 5)
    assert abs(mat.det()-np.linalg.det(npmat)) < 1e-13
    inv = mat.inv()
    npinv = np.linalg.inv(npmat)
    assert abs(inv.sum()-np.sum(npinv)) < 1e-13
    assert abs(np.sum(inv.tonumpy()-npinv)) < 1e-13
    assert abs(np.sum((inv*mat).tonumpy()-np.eye(5, 5))) < 1e-13
if __name__ == "__main__":
    test()
