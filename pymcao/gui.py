import numpy as np
from os import path
from bokeh.models import Button, Div, ColumnDataSource
from bokeh.plotting import figure
from bokeh.layouts import column, gridplot
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
            message = await self.socket.recv_pyobj()            
            self.message_callback(message)

    def send(self, message):
        @without_document_lock
        async def _send_message():
            await self.socket.send_pyobj(message)
        self.doc.add_next_tick_callback(_send_message)

class GUI(object):
    def __init__(self, curdoc):
        self.curdoc = curdoc

        self.subproc = ZmqSubProcessClient(self.curdoc(), port=5008)
        self.subproc.listen_to_computations(self.process_message)

        self.div = Div(text="Waiting for connection with simulator...")
        curdoc().add_root(column(self.div))

        self.n_rows = 5

        self.labels = ['Wavefront', 'WFS', 'Correlation', 'Measured', 'Residual']
        self.palettes = ['Viridis256', 'Greys256', 'Greys256', 'Viridis256', 'Viridis256']

    def init_gui(self, message):

        if (message['mode'] == 'MCAO'):
            self.n_stars = int(message['nstars'])            
        
        if (message['mode'] == 'SCAO'):
            self.n_stars = 1


        self.div.text = "Connected"

        d = np.zeros((100,100))

        self.p = [[None for _ in range(self.n_stars)] for _ in range(self.n_rows)]
        self.source = [[None for _ in range(self.n_stars)] for _ in range(self.n_rows)]
    
        for i in range(self.n_stars):
            for j in range(self.n_rows):

                self.source[j][i] = ColumnDataSource(data=dict(image=[d]))

                self.p[j][i] = figure(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")], plot_width=200, plot_height=200, title=self.labels[j]+f'{i}')
                self.p[j][i].x_range.range_padding = self.p[j][i].y_range.range_padding = 0

            # must give a vector of image data for image parameter
                self.p[j][i].image('image', source=self.source[j][i], x=0, y=0, dw=1, dh=1, palette=self.palettes[j])
        
        self.curdoc().add_root(gridplot(self.p))

    def process_message(self, message, doc=curdoc()):
        def show():
            for i in range(self.n_stars):
                for j in range(self.n_rows):
                    label = f'{self.labels[j]}{i}'
                    self.source[j][i].data = dict(image=[message[label]])

        def change():
            self.init_gui(message)

        if (message['action'] == 'INIT'):                
            doc.add_next_tick_callback(change)        
        else:            
            doc.add_next_tick_callback(show)


gui = GUI(curdoc)