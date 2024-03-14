from psyclone.autodiff.psyir import ADPSyIR
from psyclone.autodiff.transformations import ADTrans

class ADTransDAG(object):
    def __init__(self, source, trans, result):
        if not isinstance(source, ADPSyIR):
            raise TypeError("")
        if not isinstance(trans, ADTrans):
            raise TypeError("")
        if type(trans) is not source.transformation:
            raise TypeError("")
        if type(result) is not trans.result_type:
            raise TypeError("")
        self._source = source
        self._trans = trans
        self._result = result
        self._next = []
        self._prev = []

    @property
    def source(self):
        return self._source
    
    @property
    def trans(self):
        return self._trans
    
    @property
    def result(self):
        return self._result
    
    @property
    def next(self):
        return self._next
    
    @property
    def prev(self):
        return self._prev
    
    def new_child(self, source, trans, result, **kwargs):
        new_trans = ADTransDAG(source, trans, result)
        new_trans.log_trans_args(**kwargs)
        self.next.append(new_trans)
        new_trans.prev.append(self)
        return new_trans
    
    def log_trans_args(self, **kwargs):
        self._trans_args = kwargs