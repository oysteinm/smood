getwd()
setwd("C:/temp/ics")
dir()

library(tidyverse)

# The Index of Consumer Sentiment 
# University of Michigan, Survey Research Center
#https://data.sca.isr.umich.edu/#

ics_month <- read_csv("scaum-479.csv", col_types = cols(yyyymm = col_date(format = "%Y%m")))

ics_data <- 
  ics_month %>% 
  select(yyyymm, ics_all) %>%
  rename(date = yyyymm, ics = ics_all)

ics_data %>%
  ggplot(aes(x = date, y = ics)) +
  geom_line() + 
  ggtitle("Index of Consumer Sentiment") +
  theme_minimal() 

ics_month_id <- read_csv("AAk7MRJC.csv", col_types = cols(YYYYMM = col_date(format = "%Y%m")))



