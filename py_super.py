import os

class segmentation(object):
    def __init__(self):
        self.__report = None

    @property
    def report(self):
        print('print out the report')

class lungSeg(segmentation):
    def __init__(self):
        super(lungSeg, self).__init__()
        self.__show = None

    @property
    def show(self):
        print('this is segmentation result')

