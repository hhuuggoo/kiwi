import pathgen
import dataframe
import datalib
import hashlib
import config
import numpy as np
import scipy as sp
import scipy.signal
import quantutils as qu

ibdat= datalib.ibhdf5(config.closemat)
names=['SPY','QQQQ','IWM']
target='SPY'
tfcounter=0
#abstract parent for all tsgen classes, fieldname,daystart,dayend,numdays are mandatory, subsequent args are
#ts specific.  daystart, dayend  are positive and negative numbers that tell us what time ranges a ts generator
#can generate values for. nidx is a function that calculates the index of a given name.
#support primitive caching, so we can nest ts gens efficiently
class tsgen:
    def __init__(self,fieldname):
        self.fieldname=fieldname
        self.prevhash=""
        self.prevts=None
        self.setRequirementsAndProviders()
        self.target=False
        self.discrete=False
    def setRequirementsAndProviders(self):
        self.daystart=0
        self.dayend=0
        self.numdays=0
    def nidx(self,names,name):
        #first idx of data is time
        return names.index(name)+1
    def tsidx(self,stidx,edidx,ts):
        e=edidx
        if edidx==0:
            e=None
        return ts[stidx:e]    
    def genTS(self,olddata, data, names, globaldaystart,globaldayend, hashcode=None):
        if (self.daystart is not None and globaldaystart<self.daystart) or \
               (self.dayend is not None and globaldayend>self.dayend):
            print 'index reqs too steep'
            return Nonean
        if self.prevhash==hashcode:
            return self.prevts
        ts = self.computeTS(olddata,data,names,hashcode=hashcode)
        if hashcode is not None:
            self.prevhash=hashcode
            self.prevts=ts
        return self.tsidx(globaldaystart-self.daystart,
                          globaldayend-self.dayend, ts)
    def computeTS(self,olddata, data, names, hashcode=None):
        #most be overriden
        return None
    def setTarget(self, discrete):
        self.target=True
        self.discrete=discrete

class multitsgen(tsgen):
    def __init__(self,fieldname,tsgenlist):
        self.tsgenlist=tsgenlist
        tsgen.__init__(self,fieldname)
    def setRequirementsAndProviders(self):
        self.daystart = np.max([x.daystart for x in self.tsgenlist])
        self.dayend = np.min([x.dayend for x in self.tsgenlist])
        self.numdays = np.max([x.numdays for x in self.tsgenlist])


#raw price series:
class rawpriceTSGen(tsgen):
    def __init__(self,fieldname,name):
        self.name=name
        tsgen.__init__(self,fieldname)
    def computeTS(self,olddata, data, names, hashcode=None):
        ts = data[:,self.nidx(names,self.name)]
        return ts

#yesterdays closing prices
class ycloseTSGen(tsgen):
    def __init__(self,fieldname,name):
        self.name=name
        tsgen.__init__(self,fieldname)
    def setRequirementsAndProviders(self):
        self.daystart=0
        self.dayend=0
        self.numdays=1
    def computeTS(self,olddata, data, names, hashcode=None):
        yestdat=olddata[-1]
        yestclose=yestdat[-1,self.nidx(names,self.name)]
        justforsize = data[:,self.nidx(names,self.name)]
        ts = np.tile(yestclose,justforsize.shape)
        return ts
#moving average time series
class smoothedTSGen(tsgen):
    def __init__(self,fieldname,tschild,lag):
        self.tschild=tschild
        self.lag=lag
        tsgen.__init__(self,fieldname)
    def setRequirementsAndProviders(self):
        self.daystart=self.tschild.daystart
        self.dayend=self.tschild.dayend
        self.numdays=0
    def computeTS(self,olddata,data,names,hashcode=None):
        ts = self.tschild.genTS(olddata,data,names,self.tschild.daystart,self.tschild.dayend,hashcode=hashcode)
        smoothts = scipy.signal.lfilter(np.ones(self.lag)/float(self.lag), [1], ts)
        for c in range(self.lag):
            smoothts[c]=np.mean(ts[:c+1])
        return self.tsidx(self.daystart,self.dayend,smoothts)
    
class simpleYld(tsgen):
    def __init__(self,fieldname,tsgen1,tsgen2):
        self.tsgen1=tsgen1
        self.tsgen2=tsgen2
        tsgen.__init__(self,fieldname)
    def setRequirementsAndProviders(self):
        self.daystart=np.max((self.tsgen1.daystart, self.tsgen2.daystart))
        self.dayend=np.min((self.tsgen1.dayend,self.tsgen2.dayend))
        self.numdays=np.max((self.tsgen1.numdays,self.tsgen2.numdays))
    def computeTS(self,olddata,data,names,hashcode=None):
        if self.fieldname=="normedratio5":
            print 'here'
        ts1 = self.tsgen1.genTS(olddata,data,names,self.daystart,self.dayend,hashcode=hashcode)
        ts2 = self.tsgen2.genTS(olddata,data,names,self.daystart,self.dayend,hashcode=hashcode)
        return (ts1/ts2)

class simpleLogYld(simpleYld):
    def computeTS(self,olddata,data,names,hashcode=None):
        if self.fieldname=="normedratio5":
            print 'here'
        ts1 = self.tsgen1.genTS(olddata,data,names,self.daystart,self.dayend,hashcode=hashcode)
        ts2 = self.tsgen2.genTS(olddata,data,names,self.daystart,self.dayend,hashcode=hashcode)
        return np.log(ts1/ts2)
    
class LinearComb(multitsgen):
    def __init__(self,fieldname,tsgenlist, factors):
        self.factors=factors
        multitsgen.__init__(self,fieldname,tsgenlist)
    def computeTS(self,olddata,data,names,hashcode=None):
        tsg=self.tsgenlist[0]
        totts = self.factors[0]*tsg.genTS(olddata,data,names,self.daystart,self.dayend,hashcode=hashcode)
        if len(self.tsgenlist)>1:
            for idx,tsg in enumerate(self.tsgenlist[1:]):
                totts = totts + self.factors[1+idx] * tsg.genTS(olddata,data,names,self.daystart,self.dayend,hashcode=hashcode)
        return totts
    
class PairsLinFit(multitsgen):
    def __init__(self,fieldname,tsgenlist):
        assert len(tsgenlist)<=2, Exception("too many tsgens for pairslinfit")
        multitsgen.__init__(self,fieldname,tsgenlist)
    def computeTS(self,olddata,data,names,hashcode=None):
        ts1 = self.tsgenlist[0].genTS(olddata,data,names,self.daystart,self.dayend,hashcode=hashcode)
        ts2 = self.tsgenlist[1].genTS(olddata,data,names,self.daystart,self.dayend,hashcode=hashcode)
        datamat = np.column_stack((ts1,ts2))
        sys = qu.getLinFit(np.matrix(ts1), np.matrix(ts2))
        factor = sys[0][0,0]
        return ts1*factor - ts2
        
class StdDevGen(tsgen):
    def __init__(self,fieldname,tschild,lag):
        self.lag=lag
        self.tschild=tschild
        tsgen.__init__(self,fieldname)
    def setRequirementsAndProviders(self):
        self.daystart=self.lag+self.tschild.daystart
        self.dayend=self.tschild.dayend
        self.numdays=0
    def computeTS(self,olddata,data,names,hashcode=None):
        ts = self.tschild.genTS(olddata,data,names,self.tschild.daystart,self.tschild.dayend,hashcode=hashcode)
        myfilt = np.ones(self.lag)/float(self.lag)
        moment2 = scipy.signal.lfilter(np.ones(self.lag)/float(self.lag), [1], ts**2)
        EX2= scipy.signal.lfilter(np.ones(self.lag)/float(self.lag), [1], ts)**2
        tsstdev=(moment2-EX2)**0.5
        return self.tsidx(self.daystart,self.dayend,tsstdev)

class tradeForecast(tsgen):
    def __init__(self,fieldname,tschild,betsz,forwardwindow):
        self.tschild=tschild
        self.betsz=betsz
        self.forwardwindow=forwardwindow
        tsgen.__init__(self,fieldname)
    def setRequirementsAndProviders(self):
        self.daystart=self.tschild.daystart
        self.dayend = -self.forwardwindow + self.tschild.dayend
        self.numdays=0
    def computeTS(self,olddata,data,names,hashcode=None):
        ts = self.tschild.genTS(olddata,data,names,self.tschild.daystart,self.tschild.dayend,hashcode=hashcode)
        edidx=self.dayend
        if self.dayend==0:
            edidx=None
        outputs = np.zeros(ts[self.daystart:edidx].shape)
        for c in range(0,len(ts)-self.forwardwindow):
            endpt=np.max((c+self.forwardwindow,len(ts)))
            windowdata=ts[c:endpt]
            windowdata=windowdata-windowdata[0]
            possibleexits=np.abs(windowdata)>self.betsz
            exitidx=np.nonzero(possibleexits)[0]
            if len(exitidx)==0:
                outputs[c]=windowdata[-1]
                continue
            exitval=windowdata[exitidx[0]]
            outputs[c]=exitval
        return outputs
class tradeForecastBinary(tsgen):
    def __init__(self,fieldname,tschild,betsz,forwardwindow):
        self.tschild=tschild
        self.betsz=betsz
        self.forwardwindow=forwardwindow
        tsgen.__init__(self,fieldname)
    def setRequirementsAndProviders(self):
        self.daystart=self.tschild.daystart
        self.dayend = -self.forwardwindow + self.tschild.dayend
        self.numdays=0
    def computeTS(self,olddata,data,names,hashcode=None):
        ts = self.tschild.genTS(olddata,data,names,self.tschild.daystart,self.tschild.dayend,hashcode=hashcode)
        edidx=self.dayend
        if self.dayend==0:
            edidx=None
        outputs = np.zeros(ts[self.daystart:edidx].shape)
        for c in range(0,len(ts)-self.forwardwindow):
            endpt=np.max((c+self.forwardwindow,len(ts)))
            windowdata=ts[c:endpt]
            windowdata=windowdata-windowdata[0]
            possibleexits=np.abs(windowdata)>self.betsz
            exitidx=np.nonzero(possibleexits)[0]
            if len(exitidx)==0:
                outputs[c]=windowdata[-1]
                continue
            exitval=windowdata[exitidx[0]]
            outputs[c]=exitval
        outputs[outputs>0.0]=1.0
        outputs[outputs<=0.0]=-1.0
        return outputs

    
class paramDL:
    def __init__(self, names,tsgenlist):
        self.flds=[]
        self.daystart=[]
        self.dayend=[]
        self.numdays=[]
        self.names=names
        self.tsgenlist=tsgenlist
        for tsgen in self.tsgenlist:
            self.flds.append(tsgen.fieldname)
            self.daystart.append(tsgen.daystart)
            self.dayend.append(tsgen.dayend)
            self.numdays.append(tsgen.numdays)
        self.output=[]
        self.olddata=[]
        self.daystart=np.max(np.array(self.daystart))
        self.dayend=np.min(np.array(self.dayend))
        self.numdays=np.max(np.array(self.numdays))
        
    def getHdrs(self):
        flds=self.flds
        discreteout=[]
        target_idx=0
        for idx,tsgen in enumerate(self.tsgenlist):
            if tsgen.discrete:
                discreteout.append("d")
            else:
                discreteout.append("c")
            if tsgen.target:
               target_idx=idx
        return (target_idx, discreteout, flds)
    def loaddata(self,data):
        if len(self.olddata)==0:
            self.olddata.append(data)
        else:
            outputtslist=[]
            for tsgen in self.tsgenlist:
                outputts=tsgen.genTS(self.olddata, data, self.names, self.daystart,self.dayend)
                outputtslist.append(outputts)
            outputtsmat=np.column_stack(outputtslist)
            for c in range(outputtsmat.shape[0]):
                self.output.append(list(outputtsmat[c,:]))
    def writeoutput(self):
        return self.output

def bridge(dl,dates,names,fname=None):
    (target_idx, discreteout, flds)=dl.getHdrs()
    outputlist=[]
    if dates=='all':
        dates=range(0,len(ibdat.datelist))
    for d in dates:
        print d
        df=ibdat.gdf(d,names)
        dl.loaddata(df.data())
        output=output+dl.writeoutput()
    return (output, target_idx, discreteout, flds)

if __name__=="__main__":
    name =["OIH","XLE"]
    rawd={}
    for n in name:
        rawd[n] = rawpriceTSGen(n,n)
    smoothd={}
    lags=[5,10,20]*4
    for n in name:
        for l in lags:
           smoothd[n+str(l)]=smoothedTSGen(n+str(l), rawd[n], l)
    smoothdiff={}
    for l in lags:
        for n in name:
            signame="smoothdiff"+n+str(l)
            smoothname=n+str(l)
            smoothdiff[signame]=simpleYld(signame,rawd[n],smoothd[smoothname])
    rawratio=simpleYld('simpleYld',rawd[name[0]], rawd[name[1]])
    smoothratiod={}
    for l in lags:
        signame="".join(name)+"smoothrat"+str(l)
        smoothratiod[signame]= smoothedTSGen(signame,rawratio,l)
    varratiod={}
    for l in lags:
        signame="".join(name)+"ratiovol"+str(l)
        varratiod[signame]=StdDevGen(signame,rawratio, l)
    normratiod={}
    for l in lags:
        signame="normedratio"+str(l)
        smoothname="".join(name)+"smoothrat"+str(l)
        varname="".join(name)+"ratiovol"+str(l)
        normedratio = simpleYld(signame,smoothratiod[smoothname], varratiod[varname])
        normratiod[signame]=normedratio
    tsgenlist=rawd.values()+smoothd.values()+smoothdiff.values()+\
               smoothratiod.values()+varratiod.values()+normratiod.values()
    tf=tradeForecast("TARGET",rawratio,.05,40)
    tf.setTarget(False)
    tsgenlist.append(tf)
    pdl=paramDL(name,tsgenlist)
    bridge(pdl,range(100),name)
    
    
