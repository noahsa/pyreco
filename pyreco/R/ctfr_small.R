install.packages("FoReco")
library(FoReco)
undebug(FoReco::ctf_tools)
obj <- ctf_tools(C = matrix(c(1, 1), 1), m = 4, Sstruc = TRUE)
