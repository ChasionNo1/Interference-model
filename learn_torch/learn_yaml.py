import yaml

"""
yaml.YAMLObject用元类来注册一个构造器（也就是代码里的 init() 方法），
让你把yaml节点转为Python对象实例，用表示器（也就是代码里的 repr() 函数）来让你把Python对象转为yaml节点
"""


class Person(yaml.YAMLObject):
    yaml_tag = '!person'

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __repr__(self):
        return '%s(name=%s, age=%d)' % (self.__class__.__name__, self.name, self.age)


james = Person('James', 20)

print(yaml.dump(james))  # Python对象实例转为yaml

lily = yaml.load('!person {name: Lily, age: 19}')

print(lily)  # yaml转为Python对象实例

