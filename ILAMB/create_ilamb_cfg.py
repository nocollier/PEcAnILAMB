from netCDF4 import Dataset
import glob
import os

h1s   = ["Ecosystem and Carbon Cycle","Hydrology Cycle","Radiation and Energy Cycle"]
h2s   = {"Ecosystem and Carbon Cycle":["gpp","nee","reco"],
         "Hydrology Cycle"           :["le"],
	 "Radiation and Energy Cycle":["h","g","netrad"]}
clrs  = {"Ecosystem and Carbon Cycle":"#ECFFE6",
         "Hydrology Cycle"           :"#E6F9FF",
	 "Radiation and Energy Cycle":"#FFECE6"}
names = {"gpp"   :"Gross Primary Productivity",
         "nee"   :"Net Ecosystem Exchange",
         "reco"  :"Ecosystem Respiration",
         "le"    :"Latent Heat Flux",
         "h"     :"Sensible Heat Flux",
	 "g"     :"Soil Heat Flux",
	 "netrad":"Net Radiation"}

for h1 in h1s:    
    print("\n[h1: %s]" % h1)
    print('bgcolor = "%s"' % clrs[h1])
    for vname in h2s[h1]:
        print("\n[h2: %s]" % names[vname])
        print("variable       = %s" % vname)
        possible = [vname.upper()]
        dsetstr = ""
        for root,subdirs,files in os.walk(os.path.join(os.environ['ILAMB_ROOT'],"DATA/Ameriflux")):
            files.sort()
            for fname in files:
                if not fname.endswith(".nc"): continue
                pname = os.path.join(root,fname)
                rname = pname.replace("%s" % os.environ['ILAMB_ROOT'],"")
                dname = fname.replace(".nc","")
                if rname.startswith("/"): rname = rname[1:]
                with Dataset(pname) as dset:
                    poss = [v for v in dset.variables.keys() if v.split("_")[0] == vname.upper()]
                    if poss:
                        possible += [v for v in poss if ((v not in possible) and (v != vname) and "TEST" not in v)]
                        dsetstr  += "\n[%s]" % dname
                        dsetstr  += "\nsource = %s\n" % rname
        print("alternate_vars = %s" % (",".join(possible)))
        print("ctype          = ConfPEcAn")
        print(dsetstr)
