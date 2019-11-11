from netCDF4 import Dataset
import glob
import os


sites= ["CA-Ca1","CA-Ca3","CA-Cbo","CA-Gro","CA-Man","CA-Oas","CA-Ojp","CA-TPD","US-Bar","US-Blo","US-Bn2","US-ChR","US-Cwt","US-CZ3","US-FPe","US-Ha1","US-HBK","US-Ho1","US-LPH","US-MMS","US-MOz","US-NR1","US-Oho","US-Syv","US-UMB","US-WCr","US-xBR"]

h1s   = ["Ecosystem and Carbon Cycle","Hydrology Cycle"]
h2s   = {"Ecosystem and Carbon Cycle":["gpp","nee","reco"],
         "Hydrology Cycle"           :["le","sh"]}
clrs  = {"Ecosystem and Carbon Cycle":"#ECFFE6",
         "Hydrology Cycle"           :"#E6F9FF"}
names = {"gpp" :"Gross Primary Productivity",
         "nee" :"Net Ecosystem Exchange",
         "reco":"Ecosystem Respiration",
         "le"  :"Latent Heat",
         "sh"  :"Sensible Heat"}

for h1 in h1s:    
    print("\n[h1: %s]" % h1)
    print('bgcolor = "%s"' % clrs[h1])
    for vname in h2s[h1]:
        print("\n[h2: %s]" % names[vname])
        print("variable       = %s" % vname)
        possible = [vname.upper()]
        dsetstr = ""
        for root,subdirs,files in os.walk(os.path.join(os.environ['ILAMB_ROOT'],"DATA")):
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
                        possible += [v for v in poss if ((v not in possible) and (v != vname))]
                        dsetstr  += "\n[%s]" % dname
                        dsetstr  += "\nsource = %s\n" % rname
        print("alternate_vars = %s" % (",".join(possible)))
        print("ctype          = ConfPEcAn")
        print(dsetstr)
