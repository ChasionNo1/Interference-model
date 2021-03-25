"""
演示如何使用**kwargs
"""


class Student:
    def __int__(self, **kwargs):
        self.name = kwargs['name']
        self.age = kwargs['age']
        self.work = kwargs['work']

    def show(self):
        print(self.name + str(self.age) + self.work)


stu = Student()
stu.name = 'cairen'
stu.age = 18
stu.work = 'stu'
stu.show()
