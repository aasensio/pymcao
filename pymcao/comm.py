import socket
import logging
import zmq
import time

__all__ = ['Comm']

class ZmqSubProcess:
    def __init__(self, port=None):        
        ctx = zmq.Context.instance()
        self.socket = ctx.socket(zmq.PAIR)
        self.socket.connect("tcp://127.0.0.1:{}".format(port))        
        
    def send(self, obj):
        self.socket.send_pyobj(obj)

    def poll(self):
        return self.socket.poll(timeout=0)

    def recv(self):
        return self.socket.recv_pyobj()

class Comm(object):

    def __init__(self, simulator, cadence=1.0, event=None):

        # Logger
        self.logger = logging.getLogger("COM  ")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []

        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        self.cadence = cadence
                
        self.logger.info(f"Simulator listening on 127.0.0.1:5008...")
        
        # Start ZeroMQ async subprocess
        zsp = ZmqSubProcess(port=5008)
        count = 0

        # Send initialization
        if (simulator.operation_mode == 'mcao'):
            message = dict(action='INIT', mode='MCAO', nstars=simulator.n_stars)            
        else:
            message = dict(action='INIT', mode='SCAO', nstars=1)
            
        zsp.send(message)

        while True:

            tmp = event.wait()

            if zsp.poll() != 0:
                message = zsp.recv()
                
            self.logger.info(f"Sending new data to GUI on 127.0.0.1:5008...")            
            try:
                message = dict()
                message['action'] ='PLOT'
                for i in range(simulator.n_stars):
                    message[f'Wavefront{i}'] = simulator.wfs[i].wavefront
                    message[f'WFS{i}'] = simulator.wfs[i].wfs_image
                    message[f'Correlation{i}'] = simulator.wfs[i].correlate_image
                    message[f'Measured{i}'] = simulator.wfs[i].reconstructed_wavefront
                    message[f'Residual{i}'] = simulator.wfs[i].wavefront-simulator.wfs[i].reconstructed_wavefront
                zsp.send(message)
            except:
                pass
            count += 1
            # time.sleep(self.cadence)