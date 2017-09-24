"""
Author: Cameron Knight
Description: Utility to wrapp arrays to be mutable
"""
class InPlaceWrapper():
    """
    Mutable wrapper for a value
    """
    def __init__(self,value):
        self.value = value

    def __call__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return self.__str__()


class InPlaceArray():
    """
    Mutable Array wrapper
    """
    def __init__(self, *args):
        self.array = []
        for arg in args:
            self.array += [InPlaceWrapper(arg)]

        self.current = 0

    def __getitem__(self, key):
        return self.array[key]

    def __setitem__(self, key, value):
        self.array[key].value = value

    def __iter__(self):
        return self

    def __next__(self):
        if self.current < len(self.array):
            self.current += 1
            return self.array[self.current - 1].value
        else:
            raise StopIteration

    def __len__(self):
        len(self.array)
    """
    adding new elements
    """
    def add(self, other):
        self.array += [InPlaceWrapper(other)]

    def __add__(self, other):
        try:
            for element in other:
                self.add(element)
        except TypeError:
            print(object, "must be iteratable")

        return self

    def __str__(self):
        return '[' + ', '.join([str(i) for i in self.array]) +']'

    def __repr__(self):
        return self.__str__()

if __name__ == '__main__':
    a = InPlaceArray(*range(1,10))
    a = a + [12,13,14]
    print('a: ', a)
    b = a[0]
    print('b: ', b)
    print('adding to b')

    b.value += 10
    print('a: ', a)
    print('b: ', b)

    for i in a:
        print(i)