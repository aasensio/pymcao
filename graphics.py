from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg

class GUI(object):

    def __init__(self):
        self.app = QtGui.QApplication([])

        ## Create window with GraphicsView widget
        win = pg.GraphicsLayoutWidget()
        win.show()  ## show widget alone in its own window
        win.setWindowTitle('MCAO simulation')

        view = win.addViewBox()

        ## lock the aspect ratio so pixels are always square
        view.setAspectLocked(True)

        ## Create image item
        img = pg.ImageItem(border='w')
        view.addItem(img)

        self.app.exec_()