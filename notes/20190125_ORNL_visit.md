# PEcAn + ILAMB Coupling Plan

*Goal*: Identify core processes that impact performance across space/time.

* Focus on adding sites and running SIPNET to give a more  comprehensive view of how even one model performs and of how  suitable our methodology is across many latitudes/ecologies
  - Simple to download more sites off of AMF website and then parse with `AMFtoNetCDF.py`
  - This is low hanging fruit, although we would want to include some kind of synthesis across sites. Something like what we did for CO2 emulation. This would introduce a `h2` heading page perhaps in addition to the dataset pages that we already have in place.
* Need to add additional plots and/or metrics
  - Currently we report begin/end of season scores but no diagnostic image to help understand them.
  - Add timing of peak of season as well with accompanying figure
  - Add some wavelet/spectral analysis to illustrate what frequencies the models capture well
* Could also focus on adding more models. Eventually we definitely  want to head in this direction.
  - This could give us an idea of the constallation of *poor* to *good* models or *simple* to *complex* and help us understand tradeoffs
  - Could be difficult to make *fair* comparisons. There are different ways to run models which some may object to. Neither is it clear what is a fair notion of a initial condition, or they may not exist at all for all sites/models.
  - Important work, but also a bigger time committment
* Add relationships analysis to the current analysis
  - What we have is thought provoking, but the next question users will ask is why the results come out the way they do.
  - We could port the current relationship analysis from ILAMB into `ConfPEcAn.py` and use any output in the AMF files. Might be more telling to look at things like Bowan ratio instead of just `sh` or `le`.
  - We will want to compare relationships against met drivers which are not part of the AMF observational files (is that true?). This is not standard output from PEcAn and thus we would need to either appeal to PEcAn to add it or write some ILAMB-PEcAn interface code which could harvest the information.
  - The variable to variable analysis is a good start, but we may also like to port Jitu's multilinear analysis into what drives the beginning and ending of seasons.
* We need to think about what PEcAn outputs we want to examine. For now we are comparing versus the median from a run, but we would like to in the future expand the analysis to better understand how parameter/forcing uncertainty affects these seasonal metrics.