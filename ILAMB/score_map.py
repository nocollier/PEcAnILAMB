import matplotlib.pyplot as plt
from netCDF4 import Dataset
import pandas as pd
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

pi = np.pi

def Smooth(x0,y0,r,A=1.,C=0.1,dt=0.1,k=1.,N=1000):
    """
    d^2x/dt^2 = f - c*dx/dt

    dz/dt = f - C*z
    dx/dt = z
    """

    def _f1(x,z,f,i): return f[:,i] - C*z
    def _f2(x,z,f): return z
    
    x = x0[...] 
    y = y0[...] 
    
    a = np.zeros(x.size)
    b = np.zeros(x.size)
    
    for i in range(N):
        
        # attractive force
        n = np.asarray([x0-x,y0-y]).T
        d = np.linalg.norm(n,axis=1)
        n /= d[:,np.newaxis].clip(1e-12)
        f = n*k*d[:,np.newaxis]

        # repulsive force
        for j in range(x0.size):            
            n = np.asarray([x-x[j],y-y[j]]).T
            d = np.linalg.norm(n,axis=1)
            n /= d[:,np.newaxis].clip(1e-12)
            f += A*((1-((d/r).clip(0,1))**2)**2)[:,np.newaxis] * n

        # RK4
        k11 = dt*_f1(x,a,f,0)
        k21 = dt*_f2(x,a,f)
        k12 = dt*_f1(x+0.5*k11,a+0.5*k21,f,0)
        k22 = dt*_f2(x+0.5*k11,a+0.5*k21,f)
        k13 = dt*_f1(x+0.5*k12,a+0.5*k22,f,0)
        k23 = dt*_f2(x+0.5*k12,a+0.5*k22,f)
        k14 = dt*_f1(x+k13,a+k23,f,0)
        k24 = dt*_f2(x+k13,a+k23,f)
        x += (k11+2*k12+2*k13+k14)/6
        a += (k21+2*k22+2*k23+k24)/6

        k11 = dt*_f1(y,b,f,1)
        k21 = dt*_f2(y,b,f)
        k12 = dt*_f1(y+0.5*k11,b+0.5*k21,f,1)
        k22 = dt*_f2(y+0.5*k11,b+0.5*k21,f)
        k13 = dt*_f1(y+0.5*k12,b+0.5*k22,f,1)
        k23 = dt*_f2(y+0.5*k12,b+0.5*k22,f)
        k14 = dt*_f1(y+k13,b+k23,f,1)
        k24 = dt*_f2(y+k13,b+k23,f)
        y += (k11+2*k12+2*k13+k14)/6
        b += (k21+2*k22+2*k23+k24)/6

        #cm = plt.get_cmap('jet')
        #plt.scatter(x,y,facecolor=cm(i/N))
        
    return x,y

def RegisterCustomColormaps():
    import colorsys as cs

    # stoplight colormap
    Rd1    = [1.,0.,0.]; Rd2 = Rd1
    Yl1    = [1.,1.,0.]; Yl2 = Yl1
    Gn1    = [0.,1.,0.]; Gn2 = Gn1
    val    = 0.9
    Rd2    = cs.rgb_to_hsv(Rd2[0],Rd2[1],Rd2[2])
    Rd2    = cs.hsv_to_rgb(Rd2[0],Rd2[1],val   )
    Yl2    = cs.rgb_to_hsv(Yl2[0],Yl2[1],Yl2[2])
    Yl2    = cs.hsv_to_rgb(Yl2[0],Yl2[1],val   )
    Gn2    = cs.rgb_to_hsv(Gn2[0],Gn2[1],Gn2[2])
    Gn2    = cs.hsv_to_rgb(Gn2[0],Gn2[1],val   )
    val    = 0.75
    Rd1    = cs.rgb_to_hsv(Rd1[0],Rd1[1],Rd1[2])
    Rd1    = cs.hsv_to_rgb(Rd1[0],Rd1[1],val   )
    Yl1    = cs.rgb_to_hsv(Yl1[0],Yl1[1],Yl1[2])
    Yl1    = cs.hsv_to_rgb(Yl1[0],Yl1[1],val   )
    Gn1    = cs.rgb_to_hsv(Gn1[0],Gn1[1],Gn1[2])
    Gn1    = cs.hsv_to_rgb(Gn1[0],Gn1[1],val   )
    p      = 0
    level1 = 0.33 # 0.5
    level2 = 0.67 # 0.75
    RdYlGn = {'red':   ((0.0     , 0.0   ,Rd1[0]),
                        (level1-p, Rd2[0],Rd2[0]),
                        (level1+p, Yl1[0],Yl1[0]),
                        (level2-p, Yl2[0],Yl2[0]),
                        (level2+p, Gn1[0],Gn1[0]),
                        (1.00    , Gn2[0],  0.0)),

              'green': ((0.0     , 0.0   ,Rd1[1]),
                        (level1-p, Rd2[1],Rd2[1]),
                        (level1+p, Yl1[1],Yl1[1]),
                        (level2-p, Yl2[1],Yl2[1]),
                        (level2+p, Gn1[1],Gn1[1]),
                        (1.00    , Gn2[1],  0.0)),
              'blue':  ((0.0     , 0.0   ,Rd1[2]),
                        (level1-p, Rd2[2],Rd2[2]),
                        (level1+p, Yl1[2],Yl1[2]),
                        (level2-p, Yl2[2],Yl2[2]),
                        (level2+p, Gn1[2],Gn1[2]),
                        (1.00    , Gn2[2],  0.0))}
    plt.register_cmap(name='stoplight', data=RdYlGn)
    
RegisterCustomColormaps()

def wedge(x0,y0,r,a0,af):
    a = np.linspace(a0,af,10)
    x = np.hstack([x0,x0+r*np.cos(a)])
    y = np.hstack([y0,y0+r*np.sin(a)])
    return [[a,b] for a,b in zip(x,y)]
    

data = {'Model':[],'Site':[],'Lat':[],'Lon':[],'Field':[],'Score':[]}
vname = {'GrossPrimaryProductivity':'gpp',
         'EcosystemRespiration':'reco',
         'NetEcosystemExchange':'nee',
         'LatentHeatFlux':'le',
         'SensibleHeatFlux':'sh',
	 'SoilHeatFlux':'g',
	 'NetRadiation':'netrad'}
#for root,subdirs,files in os.walk("./_build"):
for root,subdirs,files in os.walk("/data2/ILAMB/results/CLM5/clm5_14/"):
    files = [f for f in files if "Benchmark" not in f and f.endswith(".nc")]
    for fname in files:
        pname = os.path.join(root,fname)
        site = fname.replace(".nc","").split("_")[-2]
        var = (root.split("/")[-2])
        with Dataset(pname) as dset:
            lat = dset.lat
            lon = dset.lon
            mname = dset.name
            data['Model'].append(mname)
            data['Site'].append(site)
            data['Lat'].append(lat)
            data['Lon'].append(lon)
            data['Field'].append(var)
            data['Score'].append(dset.groups["MeanState"].groups["scalars"].variables["Overall Score global"][...])

df = pd.DataFrame(data)
Vs = df['Field'].unique()
da = 2*pi/len(Vs)

cmap = plt.get_cmap('stoplight')
cmap.set_under('0.4')

sites = df['Site'].unique()
X = np.zeros(sites.size)
Y = np.zeros(X.size)
for i,site in enumerate(sites):
    sdf = df.loc[df['Site']==site]
    X[i] = sdf['Lon'].iloc[0]
    Y[i] = sdf['Lat'].iloc[0]
    if site == "US-xBR": X[i] -= 0.1
    if site == "US-LPH": X[i] -= 0.1
X,Y = Smooth(X,Y,10,dt=10,A=3e-4,C=0)
#plt.show()


xs = {}
ys = {}
for i,site in enumerate(sites):
    xs[site] = X[i]
    ys[site] = Y[i]

fig = plt.figure(figsize=(15,7.15))
ax = fig.add_subplot(position=[0.01,0.01,0.98,0.98],projection=ccrs.PlateCarree())
for site in sites:
    sdf = df.loc[df['Site']==site]
    ax.text(xs[site],ys[site]-2,site,ha='center',va='top',transform=ccrs.PlateCarree())
    for i,v in enumerate(Vs):
        try:
            s = float(sdf.loc[sdf['Field'] == v].Score)
        except:
            s = -1        
        ax.scatter(xs[site],
                   ys[site],
                   marker=wedge(0,0,1,i*da,(i+1)*da),
                   linewidth=0,
                   s=1000,
                   facecolor=cmap(s),
                   transform=ccrs.PlateCarree())


ax.set_extent([-170,-62,23.5,73],ccrs.PlateCarree())
ax.add_feature(cfeature.NaturalEarthFeature('physical','land','110m',
                                            edgecolor='face',
                                            facecolor='0.875'),zorder=-1)
ax.add_feature(cfeature.NaturalEarthFeature('physical','ocean','110m',
                                            edgecolor='face',
                                            facecolor='0.750'),zorder=-1)

cax = fig.add_axes([0.05, 0.1, 0.2, 0.05])
for i,v in enumerate(Vs):
    ax.scatter(-135,
               29,
               marker=wedge(0,0,1,i*da,(i+1)*da),
               linewidth=0,
               s=1000,
               facecolor=cmap(-1),
               transform=ccrs.PlateCarree())
    r0=2.6
    amid = (i+0.5)*da
    ax.text(-135+r0*np.sin(amid),
            +29 +r0*np.cos(amid),
            vname[v],
            ha = 'center',
            va = 'center',
            transform=ccrs.PlateCarree())
fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=cax, orientation='horizontal',label='Overall Score')
fig.savefig("overview.png")
plt.show()




