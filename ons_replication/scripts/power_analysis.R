install.packages("kappaSize")
library(kappaSize)


## How many clinical reviewers do we need?

# ...With 2 reviewers
FixedNBinary(kappa0 = 0.70,     # Assumed true k (proportion of agreement)
             n      = 144,      # Number of cases
             props  = 0.50,     # Proportion of positive cases
             raters = 2,        # Number of raters    
             alpha  = 0.05)     # Type I error rate

# >> A total of 144 subjects can produce a lower limit for kappa of 0.589.

# ...With 3 reviewers
FixedNBinary(kappa0 = 0.70,   
             n      = 144,      
             props  = 0.50,     
             raters = 3,         
             alpha  = 0.05)     

# >> A total of 144 subjects can produce a lower limit for kappa of 0.617.

# ...With 4 reviewers
FixedNBinary(kappa0 = 0.70,   
             n      = 144,      
             props  = 0.50,     
             raters = 4,         
             alpha  = 0.05)     

# >> A total of 144 subjects can produce a lower limit for kappa of 0.626.

# ...With 5 reviewers
FixedNBinary(kappa0 = 0.70,   
             n      = 144,      
             props  = 0.50,     
             raters = 5,         
             alpha  = 0.05)     

# >> A total of 144 subjects can produce a lower limit for kappa of 0.63.

