import orangedatabridge as odb
import numpy as np

def pairsgen1():
    name =["OIH","XLE"]
    rawd={}
    for n in name:
        rawd[n] = odb.rawpriceTSGen(n,n)
    smoothd={}
    lags=[20,40,80]
    for n in name:
        for l in lags:
           smoothd[n+str(l)]=odb.smoothedTSGen(n+str(l), rawd[n], l)
    smoothdiff={}
    for l in lags:
        for n in name:
            signame="smoothdiff"+n+str(l)
            smoothname=n+str(l)
            smoothdiff[signame]=odb.simpleYld(signame,rawd[n],smoothd[smoothname])
    rawratio=odb.simpleYld('simpleYld',rawd[name[0]], rawd[name[1]])
    smoothratiod={}
    for l in lags:
        signame="".join(name)+"smoothrat"+str(l)
        smoothratiod[signame]= odb.smoothedTSGen(signame,rawratio,l)
    varratiod={}
    for l in lags:
        signame="".join(name)+"ratiovol"+str(l)
        varratiod[signame]=odb.StdDevGen(signame,rawratio, l)
    normratiod={}
    for l in lags:
        signame="normedratio"+str(l)
        smoothname="".join(name)+"smoothrat"+str(l)
        varname="".join(name)+"ratiovol"+str(l)
        normedratio = odb.simpleYld(signame,smoothratiod[smoothname], varratiod[varname])
        normratiod[signame]=normedratio
    tf=odb.tradeForecastBinary("TARGET",rawratio,.005,40)
    tf.setTarget(True)
    tsgenlist=[tf] + rawd.values()+smoothd.values()+smoothdiff.values()+\
               smoothratiod.values()+varratiod.values()+normratiod.values()

    #tsgenlist=[tf]
    pdl=odb.paramDL(name,tsgenlist)
    data==odb.bridge(pdl,range(100),name)
    return data


def pairsgen2(dates):
    name =["OIH","XLE"]
    rawd={}
    for n in name:
        rawd[n] = odb.rawpriceTSGen(n,n)
    spreadgen = odb.PairsLinFit("hedgepair", [rawd[name[0]], rawd[name[1]]])
    smoothd={}
    lags=[20,40,80]
    for l in lags:
        signame="hedgepair"+str(l)
        smoothd[signame]=odb.smoothedTSGen(signame, spreadgen, l)
    smoothdiff={}
    for l in lags:
        signame="pairssmoothdiff"+str(l)
        smoothname="hedgepair"+str(l)
        smoothdiff[signame]=odb.LinearComb(signame,[smoothd[smoothname], spreadgen], [-1.0,1.0])
    vold={}
    for l in lags:
        signame="pairvol"+str(l)
        vold[signame]=odb.StdDevGen(signame,spreadgen,l)
    normd={}
    for l in lags:
        strl=str(l)
        signame="normed"+strl
        name1="pairssmoothdiff"+strl
        name2="pairvol"+strl
        normd[signame]=odb.simpleYld(signame,smoothdiff[name1], vold[name2])
    tf=odb.tradeForecastBinary("TARGET",spreadgen,0.05,40)
    tf.setTarget(True)
    tsgenlist=[tf] + normd.values()
    pdl=odb.paramDL(name,tsgenlist)
    output = odb.bridge(pdl,dates,name)
    return output
