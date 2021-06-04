import numpy as np
from os import path
from bokeh.models import Button, Div, ColumnDataSource
from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.document import without_document_lock
from bokeh.io import curdoc
import zmq
import zmq.asyncio
from bokeh.document import without_document_lock
from bokeh.application.handlers import Handler

def on_session_destroyed(self, session_context):
    if hasattr(session_context, "on_destroyed"):
        return session_context.on_destroyed(session_context)

Handler.on_session_destroyed = on_session_destroyed

class ZmqSubProcessClient:
    def __init__(self, doc, port=5006):
        ctx = zmq.asyncio.Context.instance()
        self.socket = ctx.socket(zmq.PAIR)
        if port == 0:
            port = self.socket.bind_to_random_port("tcp://127.0.0.1")
        else:
            addr = "tcp://127.0.0.1:{}".format(port)
            self.socket.bind(addr)
        self.port = port
        self.doc = doc
        
    
    def listen_to_computations(self, message_callback):                
        self.message_callback = message_callback
        self.doc.add_next_tick_callback(self.message_loop)
        self.doc.session_context.on_destroyed = self.destroy

    def destroy(self, session_context):
        self.process.kill()

    @without_document_lock
    async def message_loop(self):
        while True:
            tmp = await self.socket.recv_pyobj()
            message_type = tmp[0]
            message = tmp[1]
            print(message_type, message)
            self.message_callback(message_type, message)

    def send(self, message):
        @without_document_lock
        async def _send_message():
            await self.socket.send_pyobj(message)
        self.doc.add_next_tick_callback(_send_message)

# class GUI(object):
#     def __init__(self, curdoc):
#         self.curdoc = curdoc

#         self.subproc = ZmqSubProcessClient(self.curdoc(), port=5008)
#         self.subproc.listen_to_computations(self.process_message)

#     def init_gui(self, message):

#         n_stars = int(message)

#         N = 500
#         x = np.linspace(0, 10, N)
#         y = np.linspace(0, 10, N)
#         xx, yy = np.meshgrid(x, y)
#         d = np.sin(xx)*np.cos(yy)

#         self.source = ColumnDataSource(data=dict(image=[d]))

#         self.p = figure(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")])
#         self.p.x_range.range_padding = self.p.y_range.range_padding = 0

#         # must give a vector of image data for image parameter
#         self.p.image('image', source=self.source, x=0, y=0, dw=10, dh=10, palette="Spectral11")      


def process_message(message_type, message, doc=curdoc()):
    def show():
        global source       
        source.data = dict(image=[message])

    if (message_type == 'INIT'):
        pass
        # self.init_gui(message)
    else:
        doc.add_next_tick_callback(show)


subproc = ZmqSubProcessClient(curdoc(), port=5008)
subproc.listen_to_computations(process_message)

N = 500
x = np.linspace(0, 10, N)
y = np.linspace(0, 10, N)
xx, yy = np.meshgrid(x, y)
d = np.sin(xx)*np.cos(yy)

source = ColumnDataSource(data=dict(image=[d]))

p = figure(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")])
p.x_range.range_padding = p.y_range.range_padding = 0

# must give a vector of image data for image parameter
p.image('image', source=source, x=0, y=0, dw=10, dh=10, palette="Spectral11")      
curdoc().add_root(column(p))
