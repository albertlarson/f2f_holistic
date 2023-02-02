# f2f_holistic


the motivation behind this work is to extend <a target='_blank' href = 'https://github.com/albertlarson/f2f'>flux to flow</a>,<br>
a paper written in the spring, summer, and early fall 2022.

coming off of the defense of my thesis proposal and the <br>
submission of f2f, i elected to revisit some failed work <br>
from 2021 / early 2022, sea surface temperature. my revisit<br>
is in the repository <a target='_blank' href='https://github.com/albertlarson/aqua'>aqua</a>, named after the aqua satellite.<br>

spending more time attempting to deconvolve sst fields,<br>
this time of using L3 Aqua AMSR-E and MODIS as the<br>
input and target whilst comparing to Argo ISAS. i found<br>
again the algorithm and architecture to be wanting. the<br>
target size is just in my opinion too large. 

that said, during my written comp exam i performed a literature<br>
review of drought propagation and found a number of studies<br>
looking at sea surface temperature as a predictor of drought.

taking steps towards drought indices while trying to keep<br>
satellite data the main thing and in the wake of my failures<br>
with deconvolution, a 💡 went off in my head. in f2f, i<br>
use a shapefile to clip the four basins. take the connecticut<br>
for example. in that experiment, no consideration is made<br>
about the importance of the gulf stream or northeastern<br>
atlantic ocean continental shelf... 

here, i look at combining land and ocean hydrology... <br>
using monthly global sst fields, there is still the issue of<br>
cloud contamination. i'm using MODIS because of the better<br>
resolution and the monthly product. to correct the issues<br>
with clouds, i chop up each monthly field (looking at yukon<br>
or columbia) into smaller squares. it's called <a target='_blank' href='https://pytorch.org/docs/stable/generated/torch.nn.Fold.html'>unfolding</a>.


it's a nifty torch utility, but it can be kind of confusing.<br>
so i spent concerted time making sure i grasped unfolding<br>
and folding. while an image is unfolded, i grab every nan <br>
value and replace them with the mean of all the pixels in the<br>
unfolded square region. it has the slight downside of creating<br>
a squared effect in the folded back up image. however, i am of<br>
the opinion that having slightly blocky data that is similar to<br>
local pixels that are close to mean is better than no data at all,<br>
because of the clouds...

beyond removing nans, i take the two water quantity layers used <br>
in f2f, but this time i use GLDAS instead of NLDAS. the initial<br>
idea for this paper was to try and use the deconvolution <br>
algorithm to improve the GLDAS datasets. because of my extensive<br> 
experience and failure with this technique, after much consideration<br>
i've elected to forego this. i believe using sst, creating a <br>
continuous land / sea layer with no nans, and comparing this to just<br>
the clipped basin region values of surface and subsurface flow make <br>
for an interesting input dataset. i still use the neural net approach<br>
with streamflow prediction, increasing the number from a single<br>
station per basin in f2f to several measurements at any given time <br>
for columbia or yukon. the hypothesis is that in between the <br>
success of f2f and the failure of f2f as a deconvolution tool, <br>
we might find a better grasp of the limits of the neural network.

