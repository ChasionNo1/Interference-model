"""
演示如何使用**kwargs
"""


class Student:
    def __init__(self, **kwargs):
        self.name = kwargs['name']
        self.age = kwargs['age']
        self.work = kwargs['work']

    def show(self):
        print(self.name + str(self.age) + self.work)


class M(Student):
    def __init__(self, **kwargs):
        super(M, self).__init__(**kwargs)
        self.time = kwargs['time']

    def show2(self):
        print(self.name + self.time)


stu = Student(name='cairen', age=18, work='stu')
stu.show()

m = M(name='a', age=13, work='ggg', time='12:00')
m.show2()
