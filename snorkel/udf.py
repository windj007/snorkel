from .utils import ProgressBar
from multiprocessing import Process, Queue, JoinableQueue
from Queue import Empty
import sys


class UDF(Process):
    def __init__(self, session=None, x_queue=None, y_set=None):
        Process.__init__(self)
        self.session = session
        self.x_queue = x_queue
        self.y_set   = y_set

    def run(self):
        """
        This method is called when the UDF is run as a Process in a multiprocess setting
        The basic routine is: get from JoinableQueue, apply, put / add outputs, loop
        """
        while True:
            try:
                x = self.x_queue.get(False)
                for y in self.apply(x):
                    if self.y_set is not None:
                        self.y_set.put(y)
                    else:
                        self.session.add(y)
                self.x_queue.task_done()
            except Empty:
                break
        if self.session is not None:
            self.session.commit()
            self.session.close()
    
    def apply(self, x):
        """This function takes in an object, and returns a generator / set / list"""
        raise NotImplementedError()


class UDFRunnerMP(object):
    """Class to run UDFs in parallel using simple queue-based multiprocessing setup"""
    def __init__(self, udf_class, session=None):
        self.udf_class = udf_class
        self.udfs      = []
        self.session   = session

    def run(self, xs, parallelism=1, y_set=None):

        # Fill a JoinableQueue with input objects
        x_queue = JoinableQueue()
        for x in xs:
            x_queue.put(x)

        # Start UDF Processes
        # TODO: What session to use?
        for i in range(parallelism):
            udf = self.udf_class(session=self.session, x_queue=x_queue, y_set=y_set)
            self.udfs.append(udf)

        # Start the UDF processes, and then join on their completion
        for udf in self.udfs:
            udf.start()

        # Join on the processes all finishing!
        nU = len(self.udfs)
        for i, udf in enumerate(self.udfs):
            udf.join()
            sys.stdout.write("\r%s / %s threads done." % (i+1, nU))
        print "\n"


class UDFRunner(object):
    """Class to run a single UDF single-threaded"""
    def __init__(self, udf, session=None):
        self.udf     = udf
        self.session = session

    def run(self, xs, y_set=None, max_n=None):
        
        # Set up ProgressBar if possible
        if hasattr(xs, '__len__') or max_n is not None:
            N  = len(xs) if hasattr(xs, '__len__') else max_n
            pb = ProgressBar(N)
        else:
            N = -1
        
        # Run single-thread
        for i, x in enumerate(xs):

            # If applicable, update progress bar
            if N > 0:
                pb.bar(i)
                if i == max_n:
                    break

            # Apply the UDF and add to either the set or the session
            for y in self.udf.apply(x):
                if y_set is not None:
                    if hasattr(y_set, 'append'):
                        y_set.append(y)
                    else:
                        y_set.add(y)
                else:
                    self.session.add(y)

        # Commit
        if self.session is not None:
            self.session.commit()

        # Close the progress bar if applicable
        if N > 0:
            pb.bar(N)
            pb.close()
