from threading import Timer

class RepeatedTimer(object):
    def __init__(self, interval, function, name, *args, **kwargs):
        self._timer     = None
        self.interval   = interval
        self.function   = function
        self.args       = args
        self.kwargs     = kwargs
        self.is_running = False
        self.name       = name

        # print('init', self.name)
        self.start()

    def _run(self):
        # print('\trun', self.name)
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        # print('\t\tstart', self.name)
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    # def ():
        # return self.is_running

    def stop(self):
        # print('\t\t\tstop', self.name)
        self._timer.cancel()
        self.is_running = False