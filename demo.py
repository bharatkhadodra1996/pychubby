import matplotlib.pyplot as plt
import numpy as np
import asyncio
import urllib.request as urllib
from pychubby.actions import Smile
from pychubby.detect import LandmarkFace


def plot():
    try:
        img = plt.imread("0.jpg")
        lf = LandmarkFace.estimate(img)
        a = Smile(scale=0.09)
        new_lf, df = a.perform(lf)
        fig, ax = plt.subplots()
        new_lf.plot(show_landmarks=False)
        plt.axis('off')
        plt.savefig("output.jpg", bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        return "success"
        pass
    except Exception as e:
        return "fail"
        pass


plot()
