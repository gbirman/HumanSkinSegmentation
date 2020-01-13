
import math

class BooleanMetrics:

    def __init__(self, pred, target):
        self.TP = (pred & target).sum().item()
        self.TN = (~pred & ~target).sum().item()
        self.FP = (pred & ~target).sum().item()
        self.FN = (~pred & target).sum().item()

    def Accuracy(self):
        if self.TP + self.TN + self.FP + self.FN == 0:
            return math.nan
        return (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)
    
    def F1(self):
        if 2 * self.TP + self.FP + self.FN == 0:
            return math.nan
        return (2 * self.TP) / (2 * self.TP + self.FP + self.FN)
    
    def Precision(self):
        if self.TP + self.FP == 0:
            return math.nan
        return self.TP / (self.TP + self.FP)
    
    def Recall(self):
        if self.TP + self.FN == 0:
            return math.nan
        return self.TP / (self.TP + self.FN)

class RunningAverage:

    def __init__(self):
        self.N = 0
        self.avg = 0
    
    def add(self, val):
        if math.isnan(val):
            return
        self.avg = ((self.avg * self.N) + val) / (self.N+1)
        self.N += 1 
