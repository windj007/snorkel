from .utils import ProgressBar


class UDF(object):
    def apply(self, x):
        raise NotImplementedError()


class UDFRunner(object):
    def __init__(self, session, udf):
        self.session = session
        self.udf     = udf

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
            y = self.udf.apply(x)
            if y_set is not None:
                y_set.append(y)
            else:
                self.session.add(y)

        # Commit
        self.session.commit()

        # Close the progress bar if applicable
        if N > 0:
            pb.bar(N)
            pb.close()


# TODO: UDFRunnerMP
