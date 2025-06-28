library(tidyverse)

## Aggregate empirical data from multiple DataShop links:
# https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=5153
# https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=5549
# https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=5604

d1 <- read_delim('./data/ds5153_student_step_All_Data_7323_2024_0514_044250.txt', delim='\t') %>% select("Anon Student Id", "Problem Hierarchy", "Problem Name", "KC (Default)", "Opportunity (Default)", "KC (Single-KC)", "Opportunity (Single-KC)", "First Attempt", "Step Name")
d1$`Anon Student Id` %>% unique %>% length
d2 <- read_delim('./data/ds5549_student_step_All_Data_7820_2023_0421_140110.txt', delim='\t') %>% select("Anon Student Id", "Problem Hierarchy", "Problem Name", "KC (Default)", "Opportunity (Default)", "KC (Single-KC)", "Opportunity (Single-KC)", "First Attempt", "Step Name")
d2$`Anon Student Id` %>% unique %>% length
d3 <- read_delim('./data/ds5604_student_step_All_Data_7880_2023_0519_155815.txt', delim='\t') %>% select("Anon Student Id", "Problem Hierarchy", "Problem Name", "KC (Default)", "Opportunity (Default)", "KC (Single-KC)", "Opportunity (Single-KC)", "First Attempt", "Step Name")
d3$`Anon Student Id` %>% unique %>% length

d_out <- rbind(d2, d3) %>% 
  mutate(student_role = case_when(
    ##
    str_detect(`Problem Name`, 'Tutee') ~ 'solver',
    str_detect(`Problem Name`, 'Tutor') ~ 'tutor',
    ##
    str_detect(`Problem Name`, 'solver') ~ 'solver',
    str_detect(`Problem Name`, 'tutor') ~ 'tutor',
    str_detect(`Problem Name`, 'apta') & substr(`Problem Name`, nchar(`Problem Name`), nchar(`Problem Name`)) == 's' ~ 'solver',
    str_detect(`Problem Name`, 'apta') & substr(`Problem Name`, nchar(`Problem Name`), nchar(`Problem Name`)) == 't' ~ 'tutor',
    TRUE ~ 'single'
  )) %>% 
  filter(student_role %in% c('solver', 'single')) %>%
  select(-student_role) #%>%


d_out %>% nrow()

write_delim(d_out, './data/apta-combined.txt', delim='\t')

d_out$`KC (Default)` %>%
  str_split('~~') %>%
  unlist() %>% 
  unique() %>%
  sort()
