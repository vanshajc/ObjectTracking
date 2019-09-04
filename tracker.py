import time
import random
import math
import numpy as np
import sys


class TrackedObject(object):
    """
    State:
    [x, y, dx, dy]

    A =
    [1 0 1 0]
    [0 1 0 1]
    [0 0 1 0]
    [0 0 0 1]


    """

    # Measurement Noise
    R = np.array([[0.1, 0],
                  [0, 0.1]])

    # Process Noise
    Q = np.array([[5, 0, 0, 0],
                  [0, 5, 0, 0],
                  [0, 0, 5, 0],
                  [0, 0, 0, 5]])

    def __init__(self, bounds):
        self.w, self.h = bounds[2], bounds[3]
        self.x, self.y = bounds[0], bounds[1]
        self.id = random.randint(0, 1000)
        self.vx = 0
        self.vy = 0
        self.p = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        self.time = time.time()

    def match(self, measurement):

        adx = min(self.x + self.w/2, measurement.x + measurement.w/2) - max(self.x - self.w/2,
                                                                            measurement.x - measurement.w/2)
        ady = min(self.y + self.h/2, measurement.y + measurement.h/2) - max(self.y - self.h/2,
                                                                            measurement.y - measurement.h/2)

        if adx > 0 and ady > 0:
            return adx*ady/max(self.w*self.h, measurement.w*measurement.h)

        return 0

    def update(self, measurement):
        dt = measurement.time - self.time
        X = np.array([self.x, self.y, self.vx, self.vy])
        Y = np.array([measurement.x, measurement.y])
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1,  0],
                      [0, 0, 0,  1]])

        M = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])

        Xhat = F.dot(X)
        Phat = F.dot(self.p).dot(F.T) + TrackedObject.Q
        K = Phat.dot(M.T).dot(np.linalg.inv(M.dot(Phat).dot(M.T) + TrackedObject.R))

        Xnew = Xhat + K.dot(Y - M.dot(X))
        Pnew = (np.eye(4) - K.dot(M)).dot(Phat)

        # print(self.x, self.y, self.vx, self.vy, '->', Xnew[0], Xnew[1], Xnew[2], Xnew[3])

        self.p = Pnew
        self.x = Xnew[0]
        self.y = Xnew[1]
        self.vx = Xnew[2]
        self.vy = Xnew[3]
        self.time = measurement.time

    def predict(self):
        dt = time.time() - self.time
        return int(self.x + self.vx*dt), int(self.y + self.vy*dt)


class Tracker(object):
    AREA_THRESHOLD = 0.2

    def __init__(self):
        self._objects = []

    def add(self, rectangle):
        obj = TrackedObject(rectangle)

        updated = False

        for o in self._objects:
            # TODO: Make a better match score
            if obj.match(o) > Tracker.AREA_THRESHOLD:
                o.update(obj)
                updated = True

        if not updated:
            self._objects.append(obj)

    def predict(self):
        filtered = []
        for obj in self._objects:
            if time.time() - obj.time < 2:
                filtered.append(obj)

        self._objects = filtered
        return [x.predict() for x in self._objects]
