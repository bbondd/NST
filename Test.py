import ray

@ray.remote
class Actor(object):
    def __init__(self, number):
        self.count = 0
        self.number = number

    def increase(self):
        self.count += 1
        print('number', self.number, 'count', self.count)


def main():
    print('redis address :')
    ray.init(input())

    actors = [Actor.remote(i) for i in range(3)]
