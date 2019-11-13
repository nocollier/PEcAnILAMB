from ILAMB.Confrontation import Confrontation
from ILAMB.Variable import Variable
from ILAMB.constants import bnd_months,lbl_months
import matplotlib.pyplot as plt
import ILAMB.ilamblib as il
import ILAMB.Post as post
from netCDF4 import Dataset
import numpy as np
import cftime
import os
import glob

class NotEnoughDataInYear(Exception):
    def __str__(self): return "NotEnoughDataInYear"

def getTimeOfDailyMaximum(v):
    """
    """
    spd   = int(round(1/np.diff(v.time_bnds,axis=1).mean()))
    begin = np.argmin(v.time_bnds[:(spd-1),0] % 1)
    end   = begin+int(v.time[begin:].size/float(spd))*spd
    t     = v.time[begin:(begin+spd)] % 1
    data  = v.data[begin:end,0].reshape((-1,spd) + v.data.shape[1:])
    data  = data.mean(axis=0)[:,0]
    tmax  = t[np.argmax(data)]
    return tmax*24
    
def findSeasonalTiming(t,x):
    """Return the beginning and ending time of the season of x.

    The data x is assumed to start out relatively small, pass through
    a seasonal period of high values, and then return to small values,
    similar to a bell curve. This routine returns the times where
    these seasonal changes occur. To do this, we accumulate the
    variable x and then try to fit 3 linear segments to each seasonal
    portion. We pose a problem then that finds the breaks which
    minimizes the residual of these three best fit lines.

    Parameters
    ----------
    time: numpy.ndarray
        time array
    x: numpy.ndarray
        a single cycle of data to extract the season from

    Returns
    -------
    tbnds: numpy.ndarray
        the beginning and ending time in an array of size 2
    """
    def cost(t,y):
        from scipy.stats import linregress
        out = linregress(t,y)
        return np.sqrt((((out.slope*t+out.intercept)-y)**2).sum())
    y = x.cumsum()
    b = int(y.size/2-1)
    e = int(y.size/2+1)
    I = np.asarray(range(2,b))
    C = np.zeros(I.shape)
    for a,i in enumerate(I): C[a] = cost(t[:i],y[:i]) + cost(t[i:e],y[i:e])
    b = I[C.argmin()]
    I = np.asarray(range(e,y.size-2))
    C = np.zeros(I.shape)
    for a,i in enumerate(I): C[a] = cost(t[b:i],y[b:i]) + cost(t[i:],y[i:])
    e = I[C.argmin()]
    return t[[b,e]]

def findSeasonalCentroid(t,x):
    """Return the centroid of the season in polar and cartesian coordinates.

    Parameters
    ----------
    time: numpy.ndarray
        time array but scaled [0,2 pi]
    x: numpy.ndarray
        a single cycle of data to extract the season from

    Returns
    -------
    centroid: numpy.ndarray
        array of size 4, [r,theta,x,y]
    """
    x0 = (x*np.cos(t/365.*2*np.pi)).mean()
    y0 = (x*np.sin(t/365.*2*np.pi)).mean()
    r0 = np.sqrt(x0*x0+y0*y0)
    a0 = np.arctan2(y0,x0)
    return {"r":r0,"theta":a0,"x":x0,"y":y0}

def getDiurnalDataForGivenYear(var,year):
    """
    """

    # Get this year's data, make sure there is enough
    spd   = int(round(1/np.diff(var.time_bnds,axis=1).mean()))
    datum = cftime.date2num(cftime.datetime(year,1,1),"days since 1850-1-1")
    ind   = np.where(year==var.year)[0]
    t     = var.time     [ind]-datum
    tb    = var.time_bnds[ind]-datum
    data  = var.data     [ind,0]

    # Reshape the data
    begin  = np.argmin(tb[:(spd-1),0] % 1)
    end    = begin+int(t[begin:].size/float(spd)-1)*spd
    shift  = int(round((var.tmax-12)/(var.dt*24)))
    begin += shift; end += shift
    shp    = (-1,spd) + data.shape[1:]
    data   = data[begin:end].reshape(shp)
    t      = t   [begin:end].reshape(shp).mean(axis=1)

    # Diurnal magnitude
    mag = Variable(name = "mag%d" % year,
                   unit = var.unit,
                   time = t,
                   data = data.max(axis=1)-data.min(axis=1))

    # Some of the tower data is 'intelligently' masked which leads to
    # too much of the data being removed to use my change-detection
    # algorithm to determine season begin/end.
    mag.skip = False
    if mag.data.mask.all(): raise NotEnoughDataInYear # if year is all masked
    dmag = (mag.data.max()-mag.data.min())
    if dmag < 1e-14: raise NotEnoughDataInYear # if diurnal mag has no amplitude
    
    # Some mask out off seasons, season is taken to be all the data
    begin_day,end_day = mag.time[mag.data.mask==False][[0,-1]] # begin/end of the mask data
    if ((begin_day < 2 and end_day < 363) or
        (begin_day > 2 and end_day > 363)):
        # this is likely a dataset which is a partial year
        raise NotEnoughDataInYear
    elif (begin_day > 2 and end_day < 363):
        # this is likely a dataset that masks out off-seasons
        season = np.asarray([begin_day,end_day])
    else:
        season   = findSeasonalTiming(mag.time,mag.data)
    centroid = findSeasonalCentroid(mag.time,mag.data)
    mag.season = season
    mag.centroid = centroid

    # Mask out off season
    mask = (t<season[0])+(t>season[1])
    data = np.ma.masked_array(data,mask=mask[:,np.newaxis]*np.ones(data.shape[1],dtype=bool))

    # Mean seasonal diurnal cycle
    uncert = np.zeros((data.shape[1],2))
    for i in range(data.shape[1]):
        d = data[:,i].compressed()
        if d.size == 0: continue
        uncert[i,:] = np.percentile(d,[10,90])
    day = np.linspace(0,1,spd+1); day = 0.5*(day[:-1]+day[1:])
    with np.errstate(under='ignore',over='ignore'):
        cycle = Variable(name = "cycle%d" % year,
                         unit = var.unit,
                         time = day,
                         data = data.mean(axis=0),
                         data_bnds = uncert)

    # Mean seasonal uptake
    uptake = Variable(unit      = var.unit,
                      time      = var.time     [ind]-datum,
                      time_bnds = var.time_bnds[ind]-datum,
                      data      = var.data     [ind,0])
    uptake.data = np.ma.masked_array(uptake.data,mask=((uptake.time<season[0])+(uptake.time>season[1])))
    uptake = uptake.integrateInTime(mean=True)
    cycle.uptake = uptake.data

    # Timing of peak seasonal cycle, could be a maximum or minimum,
    # check second derivative of a best-fit parabola to the daytime
    # data.
    begin = int(spd  /4)
    end   = int(spd*3/4)
    p = np.polyfit(cycle.time[begin:end],cycle.data[begin:end],2)
    if p[0] < 0:
        cycle.peak = day[cycle.data.argmax()]*24
    else:
        cycle.peak = day[cycle.data.argmin()]*24

    return mag,cycle


def DiurnalScalars(obs_mag,mod_mag,obs_cycle,mod_cycle):
    """
    """
    score_scaling = np.log(0.1) / 0.5  # 50% relative error equals a score of .1
    obs_season_length = obs_mag.season[1]-obs_mag.season[0]
    score_sbegin = np.exp(score_scaling*np.abs(mod_mag.season[0]-obs_mag.season[0])/obs_season_length)
    score_send   = np.exp(score_scaling*np.abs(mod_mag.season[1]-obs_mag.season[1])/obs_season_length)
    obs_centroid = np.linalg.norm([obs_mag.centroid['x'],obs_mag.centroid['y']])
    score_centroid = np.linalg.norm([mod_mag.centroid['x']-obs_mag.centroid['x'],
                                     mod_mag.centroid['y']-obs_mag.centroid['y']]) / obs_centroid
    score_centroid = np.exp(-score_centroid)
    score_peak = np.abs(mod_cycle.peak-obs_cycle.peak)/12.
    score_peak = 1-(score_peak>1)*(1-score_peak)-(score_peak<=1)*score_peak
    score_uptake = 0
    with np.errstate(under='ignore',over='ignore'):
        score_uptake = np.exp(-np.abs((mod_cycle.uptake-obs_cycle.uptake)/obs_cycle.uptake))
    #print(obs_cycle.uptake,mod_cycle.uptake,score_uptake)
    return score_sbegin,score_send,score_centroid,score_peak,score_uptake

class ConfPEcAn(Confrontation):
    """A confrontation for examining the diurnal
    """
    def __init__(self,**keywords):

        # Calls the regular constructor
        super(ConfPEcAn,self).__init__(**keywords)

        obs = Variable(filename       = self.source,
                       variable_name  = self.variable,
                       alternate_vars = self.alternate_vars,
                       convert_calendar = False)
        self.years = np.asarray([t.year for t in cftime.num2date(obs.time,"days since 1850-1-1")],dtype=int)
        self.years = np.unique(self.years)

        # Setup a html layout for generating web views of the results
        pages = []

        # Mean State page
        pages.append(post.HtmlPage("MeanState","Mean State"))
        pages[-1].setHeader("CNAME / RNAME / MNAME")
        pages[-1].setSections(["Seasonal Diurnal Cycle",]+["%d" % y for y in self.years])
        pages.append(post.HtmlAllModelsPage("AllModels","All Models"))
        pages[-1].setHeader("CNAME / RNAME")
        pages[-1].setSections([])
        pages[-1].setRegions(self.regions)
        pages.append(post.HtmlPage("DataInformation","Data Information"))
        pages[-1].setSections([])
        pages[-1].text = "\n"
        with Dataset(self.source) as dset:
            for attr in dset.ncattrs():
                a = dset.getncattr(attr)
                if 'astype' in dir(a): a = a.astype('str')
                if 'encode' in dir(a): a = a.encode('ascii','ignore')
                pages[-1].text += "<p><b>&nbsp;&nbsp;%s:&nbsp;</b>%s</p>\n" % (attr,a)
        self.layout = post.HtmlLayout(pages,self.longname)

    def stageData(self,m):

        obs = Variable(filename       = self.source,
                       variable_name  = self.variable,
                       alternate_vars = self.alternate_vars,
                       convert_calendar = False)
        if obs.time is None: raise il.NotTemporalVariable()
        self.pruneRegions(obs)

        # Try to extract a commensurate quantity from the model
        mod = m.extractTimeSeries(self.variable,
                                  alt_vars     = self.alternate_vars,
                                  expression   = self.derived,
                                  convert_calendar = False,
                                  lats         = None if obs.spatial else obs.lat,
                                  lons         = None if obs.spatial else obs.lon)

        # When we make things comparable, sites can get pruned, we
        # also need to prune the site labels
        lat = np.copy(obs.lat); lon = np.copy(obs.lon)
        obs,mod = il.MakeComparable(obs,mod,clip_ref=False,prune_sites=True,allow_diff_times=True)

        # Some datasets / models return data in UTC, others are local
        # time. Try to correct by looking at the time of maximum
        # incident radiation.
        try:
            inc = Variable(filename       = self.source,
                           variable_name  = "LW_IN",
                           convert_calendar = False)
            obs.tmax = getTimeOfDailyMaximum(inc)
        except:
            obs.tmax = 12.
        try:
            inc = m.extractTimeSeries("FSDS",
                                      convert_calendar = False,
                                      lats = None if obs.spatial else obs.lat,
                                      lons = None if obs.spatial else obs.lon)
            mod.tmax = getTimeOfDailyMaximum(inc)
        except:
            mod.tmax = 12.
        
        return obs,mod

    def confront(self,m):

        # Grab the data
        obs,mod = self.stageData(m)

        # What years does the analysis run over?
        obs.year = np.asarray([t.year for t in cftime.num2date(obs.time,"days since 1850-1-1")],dtype=int)
        mod.year = np.asarray([t.year for t in cftime.num2date(mod.time,"days since 1850-1-1")],dtype=int)
        
        # Analysis
        mod_file = os.path.join(self.output_path,"%s_%s.nc"        % (self.name,m.name))
        obs_file = os.path.join(self.output_path,"%s_Benchmark.nc" % (self.name,      ))
        with il.FileContextManager(self.master,mod_file,obs_file) as fcm:

            # Encode some names and colors
            fcm.mod_dset.setncatts({"name" :m.name,
                                    "color":m.color,
                                    "complete":0})
            if self.master:
                fcm.obs_dset.setncatts({"name" :"Benchmark",
                                        "color":np.asarray([0.5,0.5,0.5]),
                                        "complete":0})

            Osbegin = []; Osend = []; Oslen = []; Opeak = []; Ouptake = []
            Msbegin = []; Msend = []; Mslen = []; Mpeak = []; Muptake = []
            Ssbegin = []; Ssend = []; Scentroid = []; Speak = []; Suptake = []
            obs_years = 0; mod_years = 0
            for y in self.years:
                
                # First try to get the obs for this year, might not
                # have enough info in which case we skip the year.
                try:
                    obs_mag,obs_cycle = getDiurnalDataForGivenYear(obs,y)
                except NotEnoughDataInYear:
                    continue
                
                # Output what we must even if the model doesn't have
                # data here.
                obs_years += 1
                Osbegin.append(obs_mag.season[0])
                Osend  .append(obs_mag.season[1])
                Oslen  .append(obs_mag.season[1]-obs_mag.season[0])
                Opeak  .append(obs_cycle.peak)
                Ouptake.append(obs_cycle.uptake)
                if self.master:
                    obs_mag.toNetCDF4(fcm.obs_dset,group="MeanState",
                                      attributes={"sbegin":obs_mag.season[0],
                                                  "send":obs_mag.season[1]})
                    obs_cycle.toNetCDF4(fcm.obs_dset,group="MeanState",
                                        attributes={"uptake":obs_cycle.uptake,
                                                    "peak":obs_cycle.peak})

                # Try to get enough data from the model to operate
                try:
                    mod_mag,mod_cycle = getDiurnalDataForGivenYear(mod,y)
                except NotEnoughDataInYear:
                    continue
                
                mod_years += 1
                Msbegin.append(mod_mag.season[0])
                Msend  .append(mod_mag.season[1])
                Mslen  .append(mod_mag.season[1]-mod_mag.season[0])
                Mpeak  .append(mod_cycle.peak)
                Muptake.append(mod_cycle.uptake)
                mod_mag.toNetCDF4(fcm.mod_dset,group="MeanState",
                                  attributes={"sbegin":mod_mag.season[0],
                                              "send":mod_mag.season[1]})
                mod_cycle.toNetCDF4(fcm.mod_dset,group="MeanState",
                                    attributes={"uptake":mod_cycle.uptake,
                                                "peak":mod_cycle.peak})

                # Get scores for this year
                ssbegin,ssend,scentroid,speak,suptake = DiurnalScalars(obs_mag,mod_mag,obs_cycle,mod_cycle)
                Ssbegin.append(ssbegin); Ssend.append(ssend)
                Scentroid.append(scentroid); Speak.append(speak)
                Suptake.append(suptake)

            # Output mean scores/scalars
            if self.master and obs_years > 0:
                Variable(name = "Number of Years global",unit="1",
                         data = obs_years).toNetCDF4(fcm.obs_dset,group="MeanState")
                Variable(name = "Computed UTC Shift global",unit="h",
                         data = obs.tmax-12).toNetCDF4(fcm.obs_dset,group="MeanState")
                Variable(name = "Season Beginning global",unit="d",
                         data = np.asarray(Osbegin).mean()).toNetCDF4(fcm.obs_dset,group="MeanState")
                Variable(name = "Season Ending global",unit="d",
                         data = np.asarray(Osend).mean()).toNetCDF4(fcm.obs_dset,group="MeanState")
                Variable(name = "Season Length global",unit="d",
                         data = np.asarray(Oslen).mean()).toNetCDF4(fcm.obs_dset,group="MeanState")
                Variable(name = "Diurnal Peak Timing global",unit="h",
                         data = np.asarray(Opeak).mean()).toNetCDF4(fcm.obs_dset,group="MeanState")
                Variable(name = "Mean Season Uptake global",unit=obs.unit,
                         data = np.asarray(Ouptake).mean()).toNetCDF4(fcm.obs_dset,group="MeanState")
            if mod_years > 0 :
                Variable(name = "Number of Years global",unit="1",
                         data = mod_years).toNetCDF4(fcm.mod_dset,group="MeanState")
                Variable(name = "Computed UTC Shift global",unit="h",
                         data = mod.tmax-12).toNetCDF4(fcm.mod_dset,group="MeanState")
                Variable(name = "Season Beginning global",unit="d",
                         data = np.asarray(Msbegin).mean()).toNetCDF4(fcm.mod_dset,group="MeanState")
                Variable(name = "Season Ending global",unit="d",
                         data = np.asarray(Msend).mean()).toNetCDF4(fcm.mod_dset,group="MeanState")
                Variable(name = "Season Length global",unit="d",
                         data = np.asarray(Mslen).mean()).toNetCDF4(fcm.mod_dset,group="MeanState")
                Variable(name = "Diurnal Peak Timing global",unit="h",
                         data = np.asarray(Mpeak).mean()).toNetCDF4(fcm.mod_dset,group="MeanState")
                Variable(name = "Mean Season Uptake global",unit=mod.unit,
                         data = np.asarray(Muptake).mean()).toNetCDF4(fcm.mod_dset,group="MeanState")
                Variable(name = "Season Beginning Score global",unit="1",
                         data = np.asarray(Ssbegin).mean()).toNetCDF4(fcm.mod_dset,group="MeanState")
                Variable(name = "Season Ending Score global",unit="1",
                         data = np.asarray(Ssend).mean()).toNetCDF4(fcm.mod_dset,group="MeanState")
                Variable(name = "Season Strength Score global",unit="1",
                         data = np.asarray(Scentroid).mean()).toNetCDF4(fcm.mod_dset,group="MeanState")
                Variable(name = "Diurnal Peak Timing Score global",unit="1",
                         data = np.asarray(Speak).mean()).toNetCDF4(fcm.mod_dset,group="MeanState")
                Variable(name = "Diurnal Uptake Score global",unit="1",
                         data = np.asarray(Suptake).mean()).toNetCDF4(fcm.mod_dset,group="MeanState")

            # Flag complete
            fcm.mod_dset.complete = 1
            if self.master: fcm.obs_dset.complete = 1

    def determinePlotLimits(self):

        self.limits = {}
        self.limits["mag"] = 0.
        self.limits["cycle"] = {}
        self.limits["cycle"]["min"] = +1e20
        self.limits["cycle"]["max"] = -1e20
        for fname in glob.glob(os.path.join(self.output_path,"*.nc")):
            with Dataset(fname) as dataset:
                if "MeanState" not in dataset.groups: continue
                group     = dataset.groups["MeanState"]
                variables = [v for v in group.variables.keys() if v not in group.dimensions.keys()]
                for vname in variables:
                    if "mag" in vname:
                        self.limits["mag"] = max(self.limits["mag"],group.variables[vname].up99)
                    if "cycle" in vname and "_bnds" in vname:
                        self.limits["cycle"]["min"] = min(self.limits["cycle"]["min"],group.variables[vname][...].min())
                        self.limits["cycle"]["max"] = max(self.limits["cycle"]["max"],group.variables[vname][...].max())

    def modelPlots(self,m):

        bname  = "%s/%s_Benchmark.nc" % (self.output_path,self.name)
        fname  = "%s/%s_%s.nc" % (self.output_path,self.name,m.name)
        if not os.path.isfile(bname): return
        if not os.path.isfile(fname): return

        # get the HTML page
        page = [page for page in self.layout.pages if "MeanState" in page.name][0]
        page.priority = ["Beginning","Ending","Strength","Score","Overall"]

        for y in self.years:

            # ---------------------------------------------------------------- #

            plt.figure(figsize=(5,5),tight_layout=True)
            has_data = False
            for name,color,alpha in zip([bname,fname],['k',m.color],[0.6,1.0]):
                try:
                    v = Variable(filename=name,variable_name="mag%d" % y,groupname="MeanState")
                    has_data = True
                except:
                    continue
                plt.polar(v.time/365.*2*np.pi,v.data,'-',color=color,alpha=alpha,lw=2)

            if has_data:
                plt.xticks(bnd_months[:-1]/365.*2*np.pi,lbl_months)
                plt.ylim(0,self.limits["mag"])
                plt.savefig("%s/%s_mag%d.png" % (self.output_path,m.name,y))
                page.addFigure("%d" % y,
                               "mag%d" % y,
                               "MNAME_mag%d.png" % y,
                               side = "DIURNAL MAGNITUDE",
                               legend = False)
            plt.close()

            # ---------------------------------------------------------------- #

            fig,ax = plt.subplots(figsize=(8,5),tight_layout=True)
            has_data = False
            unit = ""
            for name,color,alpha,lbl in zip([bname,fname],['k',m.color],[0.6,1.0],['Benchmark',m.name]):
                try:
                    v = Variable(filename=name,variable_name="cycle%d" % y,groupname="MeanState")
                    has_data = True
                    unit = v.unit
                except:
                    continue
                v.plot(ax,color=color,alpha=alpha,lw=2,label=lbl)
            if has_data:
                ax.set_xticks(np.linspace(0,1,9)/365+1850)
                ax.set_xticklabels(["%2d:00" % t for t in np.linspace(0,24,9)])
                ax.set_ylim(self.limits['cycle']['min'],self.limits['cycle']['max'])
                ax.grid(True)
                ax.set_ylabel(post.UnitStringToMatplotlib(unit))
                ax.set_xlabel("local time")
                ax.legend(bbox_to_anchor=(0,1.005,1,0.25),loc='lower left',mode='expand',ncol=2,borderaxespad=0,frameon=False)
                plt.savefig("%s/%s_cycle%d.png" % (self.output_path,m.name,y))
                page.addFigure("%d" % y,
                               "cycle%d" % y,
                               "MNAME_cycle%d.png" % y,
                               side = "SEASONAL DIURNAL CYCLE",
                               legend = False)
            plt.close()
