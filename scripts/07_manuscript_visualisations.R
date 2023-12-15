#### Master Script 07: Visualise study results for manuscript ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# Visualise the distribution and transitions of TILBasic over days of ICU stay
# Visualise the distribution of TILBasic over days of ICU stay
# Visualise the distribution of changes in TILBasic over days of ICU stay
# Visualise the distribution of next-day TILBasic given previous-day TILBasic
# Visualise the distribution of previous-day and next-day TILBasic given a transition occurred
# Visualise feature TimeSHAP values across all points of TILBasic transition 
# Visualise feature TimeSHAP at all points of TILBasic transition for each starting TILBasic value
# Visualise missing feature TimeSHAP at all points of TILBasic transition
# Visualise threshold-level AUC in prediction of next-day TILBasic
# Visualise threshold-level calibration slope in prediction of next-day TILBasic
# Visualise threshold-level calibration curves in prediction of next-day TILBasic

### I. Initialisation
# Import necessary libraries
library(tidyverse)
library(readxl)
library(plotly)
library(ggbeeswarm)
library(cowplot)
library(rvg)
library(svglite)
library(openxlsx)
library(gridExtra)
library(extrafont)
library(survminer)
library(survival)
library(ggalluvial)

# Import custom plotting functions
source('functions/plotting.R')

## Define parameters for manuscript visualisations
# Define days of ICU stay to focus on for TIL
TIL.assessment.days <- c(1:7,10,14,21,28)

# Define subset of TIL assessment days for study
study.TIL.days <- c(1:7,10,14)

# Create list of ICU stay day labels based on defined days of focus
TIL.day.labels <- paste('Day',TIL.assessment.days)

# Define colour keys based on hex values
BluRedDiv5 <- c('#003f5c','#8386b2','#ffd4ff','#f68cba','#de425b')
StrongBluRedDiv5 <- c('#003f5c','#6b7ab6','#ecb0ff','#f376b5','#de425b')
BluRedDiv3 <- c('#003f5c','#ffd4ff','#de425b')
StrongBluRedDiv3 <- c('#003f5c','#ecb0ff','#de425b')
Palette4 <- c('#003f5c','#7a5195','#ef5675','#ffa600')
Palette5 <- c('#003f5c','#58508d','#bc5090','#ff6361','#ffa600')

### Visualise the distribution and transitions of TILBasic over days of ICU stay
## Load and prepare dataframes
# Call function to get formatted TILBasic values over days of ICU stay
study.days.TILBasic <- get.formatted.TILBasic(study.TIL.days) %>%
  mutate(ICUDay=fct_reorder(factor(ICUDay), TILTimepoint),
         TILBasic = factor(TILBasic,levels=c('4','3','2','1','0','Missing','Discharged','WLST or Died')))

# Format long dataframe into summarised form for alluvial plotting
study.days.TILBasic.counts <- study.days.TILBasic %>%
  pivot_wider(id_cols = GUPI, names_from = ICUDay, values_from = TILBasic) %>%
  group_by(across(c(-GUPI))) %>%
  summarise(Freq = n()) %>%
  ungroup() %>%
  mutate(SampleId = row_number()) %>%
  pivot_longer(cols = -c(Freq,SampleId),
               names_to = 'ICUDay',
               values_to = 'TILBasic') %>%
  mutate(ICUDay = factor(ICUDay,levels=paste('Day',study.TIL.days)),
         TILBasic = factor(TILBasic,levels=c('4','3','2','1','0','Missing','Discharged','WLST or Died')),
         TILTimepoint = as.integer(word(ICUDay,2)),
         MapPoint = case_when(TILTimepoint<=7 ~ TILTimepoint,
                              TILTimepoint==10 ~ 8.25,
                              TILTimepoint==14 ~ 9.5))

## Plot and save TILBasic alluvial plot over days of ICU stay
# Create ggplot object of TILBasic alluvial plot
TILBasic.alluvial.plot <- ggplot(study.days.TILBasic.counts,
                                 aes(x = MapPoint, stratum = TILBasic, alluvium = SampleId, y = Freq, fill = TILBasic)) +
  scale_x_continuous(breaks = c(1:7,8.25,9.5),
                     labels = study.TIL.days,
                     expand = c(0,0)) +
  scale_y_continuous(expand = c(0.01,0.01)) +
  geom_flow() +
  geom_stratum(size=1/.pt,width = (5/12)) +
  xlab('Day of ICU stay') +
  ylab('Count (n)') + 
  scale_fill_manual(values = c(rev(StrongBluRedDiv5),'gray60','gray40','gray20')) +
  guides(fill=guide_legend(title="TIL(Basic)",nrow = 1,reverse = T)) +
  geom_text(stat = "stratum",
            aes(label = scales::percent(after_stat(prop), accuracy = 1)),
            size=6/.pt,
            family = 'Roboto Condensed',
            color='white',
            min.y = 25) +
  theme_minimal(base_family = 'Roboto Condensed') +
  theme(strip.text = element_blank(),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_blank(),
        panel.border = element_blank(),
        panel.spacing = unit(10, 'points'),
        axis.text.x = element_text(size = 6, color = "black",margin = margin(0,0,0,0)),
        axis.text.y = element_text(size = 6, color = "black",margin = margin(0,0,0,0)),
        axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
        axis.title.y = element_text(size = 7, color = "black",face = 'bold',margin = margin(0,0,0,0)),
        legend.position = 'bottom',
        legend.title = element_text(size = 7, color = "black", face = 'bold'),
        legend.text=element_text(size=6),
        legend.key.size = unit(1.3/.pt,"line")
  )

# Create directory for current date and save TILBasic distribution over days of ICU stay
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'TIL_Basic_alluvial.svg'),TILBasic.alluvial.plot,device=svglite,units='in',dpi=600,width=3.75,height=3.81)

### Visualise the distribution of TILBasic over days of ICU stay
## Load and prepare dataframes
# Call function to get formatted TILBasic values over days of ICU stay
study.days.TILBasic <- get.formatted.TILBasic(study.TIL.days) %>%
  mutate(ICUDay=fct_reorder(factor(ICUDay), TILTimepoint),
         TILBasic = factor(TILBasic,levels=c('4','3','2','1','0','Missing','Discharged','WLST or Died')),
         Grouping = case_when(TILTimepoint<=6~'1',
                              TILTimepoint<=9~'2',
                              TILTimepoint<=13~'3',
                              TILTimepoint<=20~'4',
                              TILTimepoint<=27~'5'))

# Add tomorrow's TILBasic values to dataframe
trans.TILBasic <- study.days.TILBasic %>% 
  left_join(study.days.TILBasic %>%
  mutate(NextTILTimepoint = TILTimepoint,
         TILTimepoint = case_when(TILTimepoint<=7 ~ TILTimepoint-1,
                                  TILTimepoint==10 ~ 7,
                                  TILTimepoint==14 ~ 10)) %>%
  rename(TomorrowTILBasic = TILBasic) %>%
  select(-c(ICUDay,Grouping))) %>%
  filter(TILTimepoint!=14) %>%
  mutate(Transition = paste(TILTimepoint,'→',NextTILTimepoint))
  
  
  # Load and clean dataframe containing ICU daily windows of study participants
  study.windows <- read.csv('../../center_tbi/CENTER-TBI/FormattedTIL/study_window_timestamps_outcomes.csv',na.strings = c("NA","NaN","", " ")) %>%
  select(GUPI,WindowIdx,WindowTotal,TimeStampStart,TimeStampEnd) %>%
  rename(TILTimepoint = WindowIdx) %>%
  mutate(TimeStampStart = as.POSIXct(TimeStampStart,format = '%Y-%m-%d',tz = 'GMT'),
         TimeStampEnd = as.POSIXct(TimeStampEnd,format = '%Y-%m-%d %H:%M:%S',tz = 'GMT'),
         TILDate = as.Date(TimeStampStart,tz = 'GMT'))

# Load and clean dataframe containing formatted TIL scores
formatted.TIL.values <- read.csv('../../center_tbi/CENTER-TBI/FormattedTIL/formatted_TIL_values.csv',na.strings = c("NA","NaN","", " ")) %>%
  select(GUPI,TILTimepoint,TILDate,TILBasic) %>%
  mutate(TILDate = as.Date(as.POSIXct(TILDate,format = '%Y-%m-%d',tz = 'GMT'),tz = 'GMT'))

# Merge study window and TILBasic information and format days
study.days.TILBasic <- study.windows %>%
  full_join(formatted.TIL.values) %>%
  filter(TILTimepoint %in% TIL.assessment.days) %>%
  mutate(ICUDay = sprintf('Day %.0f',TILTimepoint),
         Grouping = case_when(TILTimepoint<=7~'1',
                              TILTimepoint<=10~'2',
                              TILTimepoint<=14~'3',
                              TILTimepoint<=21~'4',
                              TILTimepoint<=28~'5'),
         ICUDay=fct_reorder(factor(ICUDay), TILTimepoint),
         TILBasic = case_when(TILBasic==4~'4',
                              TILBasic==3~'3',
                              TILBasic==2~'2',
                              TILBasic==1~'1',
                              TILBasic==0~'0',
                              is.na(TILBasic)~'Missing'),
         TILBasic = factor(TILBasic,levels=c('0','1','2','3','4','Missing'))) %>%
  count(ICUDay, Grouping, TILBasic) %>%
  group_by(ICUDay, Grouping) %>%
  mutate(pct=100*(n/sum(n)),
         Label = paste0(as.character(signif(pct,2)),'%'))

## Plot and save TILBasic distribution over days of ICU stay
# Create ggplot object of TILBasic distribution
TILBasic.distributions <- study.days.TILBasic %>%
  ggplot(aes(fill=fct_rev(TILBasic), y=n, x=ICUDay)) + 
  geom_bar(position="stack", stat="identity") +
  geom_text(aes(label = Label),
            position = position_stack(vjust = .5),
            size=6/.pt,
            family = 'Roboto Condensed') +
  scale_fill_manual(values=rev(c(BluRedDiv5,'lightgray'))) +
  guides(fill=guide_legend(title="TIL(Basic)",nrow = 1,reverse = T)) +
  scale_y_continuous(expand = expansion(mult = c(.00, .00)))+
  theme_minimal(base_family = 'Roboto Condensed') +
  ylab('Count (n)') +
  xlab('Day of ICU stay') +
  facet_grid(cols = vars(Grouping), scales = 'free_x', switch = 'x', space = 'free_x') +
  theme(
    strip.text = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_blank(),
    panel.border = element_blank(),
    panel.spacing = unit(10, 'points'),
    axis.text.x = element_text(size = 6, color = "black",margin = margin(0,0,0,0)),
    axis.text.y = element_text(size = 6, color = "black",margin = margin(0,0,0,0)),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold',margin = margin(0,0,0,0)),
    legend.position = 'bottom',
    legend.title = element_text(size = 7, color = "black", face = 'bold'),
    legend.text=element_text(size=6),
    legend.key.size = unit(1.3/.pt,"line")
  )

# Create directory for current date and save TILBasic distribution over days of ICU stay
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'TIL_Basic_distributions_over_time.svg'),TILBasic.distributions,device=svglite,units='in',dpi=600,width=3.75,height=3.81)

### Visualise the distribution of changes in TILBasic over days of ICU stay
## Load and prepare dataframes
# Call function to get formatted TILBasic values over days of ICU stay
study.days.TILBasic <- get.formatted.TILBasic(study.TIL.days) %>%
  mutate(ICUDay=fct_reorder(factor(ICUDay), TILTimepoint),
         TILBasic = factor(TILBasic,levels=c('4','3','2','1','0','Missing','Discharged','WLST or Died')),
         Grouping = case_when(TILTimepoint<=6~'1',
                              TILTimepoint<=9~'2',
                              TILTimepoint<=13~'3',
                              TILTimepoint<=20~'4',
                              TILTimepoint<=27~'5'))

# Add tomorrow's TILBasic values to dataframe
trans.TILBasic <- study.days.TILBasic %>% 
  left_join(study.days.TILBasic %>%
              mutate(NextTILTimepoint = TILTimepoint,
                     TILTimepoint = case_when(TILTimepoint<=7 ~ TILTimepoint-1,
                                              TILTimepoint==10 ~ 7,
                                              TILTimepoint==14 ~ 10)) %>%
              rename(TomorrowTILBasic = TILBasic) %>%
              select(-c(ICUDay,Grouping))) %>%
  filter(TILTimepoint!=14,
         !(TILBasic %in% c('WLST or Died','Discharged')),
         !(TomorrowTILBasic %in% c('WLST or Died','Discharged'))) %>%
  mutate(TimepointTransition = paste(TILTimepoint,'→',NextTILTimepoint),
         TILBasicTransition = case_when(TILBasic == 'Missing' ~ 'Missing',
                                        TomorrowTILBasic == 'Missing' ~ 'Missing',
                                        TILBasic == TomorrowTILBasic ~ 'No change',
                                        as.character(TomorrowTILBasic)>as.character(TILBasic)~'Increase',
                                        as.character(TomorrowTILBasic)<as.character(TILBasic)~'Decrease'),
         TILBasicTransition = factor(TILBasicTransition,levels=c('Missing','Decrease','No change','Increase'))) %>%
  count(TimepointTransition, Grouping, TILBasicTransition) %>%
  group_by(TimepointTransition, Grouping) %>%
  mutate(pct=100*(n/sum(n)),
         TransTotal = sum(n),
         Label = sprintf('%.0f%%',pct))

## Plot and save change in TILBasic distribution over days of ICU stay
# Create ggplot object of daily changes in TILBasic distribution
dTILBasic.distributions <- trans.TILBasic %>%
  ggplot(aes(fill=fct_rev(TILBasicTransition), y=n, x=TimepointTransition)) + 
  geom_bar(position="stack", stat="identity",color='black',size=1/.pt) +
  geom_text(aes(label = Label),
            position = position_stack(vjust = .5),
            size=6/.pt,
            family = 'Roboto Condensed',
            color='white') +
  scale_fill_manual(values=rev(c('gray60',BluRedDiv3))) +
  guides(fill=guide_legend(title="Change in TIL(Basic)",nrow = 1,reverse = T)) +
  scale_y_continuous(expand = expansion(mult = c(.00, .00)))+
  scale_x_discrete(expand = expansion(mult = c(.00, .00)))+
  theme_minimal(base_family = 'Roboto Condensed') +
  ylab('Count (n)') +
  xlab('Day-to-day steps in ICU stay') +
  facet_grid(cols = vars(Grouping), scales = 'free_x', switch = 'x', space = 'free_x') +
  theme(
    strip.text = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_blank(),
    panel.border = element_blank(),
    panel.spacing = unit(10, 'points'),
    axis.text.x = element_text(size = 6, color = "black",margin = margin(0,0,0,0)),
    axis.text.y = element_text(size = 6, color = "black",margin = margin(0,0,0,0)),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold',margin = margin(0,0,0,0)),
    legend.position = 'bottom',
    legend.title = element_text(size = 7, color = "black", face = 'bold'),
    legend.text=element_text(size=6),
    legend.key.size = unit(1.3/.pt,"line")
  )

# Create directory for current date and save change in TILBasic distribution over days of ICU stay
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'change_in_TIL_Basic_distributions_over_time.svg'),dTILBasic.distributions,device=svglite,units='in',dpi=600,width=3.75,height=3.81)

### Visualise the distribution of next-day TILBasic given previous-day TILBasic
## Load and prepare dataframes
# Load and clean dataframe containing ICU daily windows of study participants
study.windows <- read.csv('../../center_tbi/CENTER-TBI/FormattedTIL/study_window_timestamps_outcomes.csv',na.strings = c("NA","NaN","", " ")) %>%
  select(GUPI,WindowIdx,WindowTotal,TimeStampStart,TimeStampEnd) %>%
  rename(TILTimepoint = WindowIdx) %>%
  mutate(TimeStampStart = as.POSIXct(TimeStampStart,format = '%Y-%m-%d',tz = 'GMT'),
         TimeStampEnd = as.POSIXct(TimeStampEnd,format = '%Y-%m-%d %H:%M:%S',tz = 'GMT'),
         TILDate = as.Date(TimeStampStart,tz = 'GMT'))

# Load and clean dataframe containing formatted TIL scores
formatted.TIL.values <- read.csv('../../center_tbi/CENTER-TBI/FormattedTIL/formatted_TIL_values.csv',na.strings = c("NA","NaN","", " ")) %>%
  select(GUPI,TILTimepoint,TILDate,TILBasic) %>%
  mutate(TILDate = as.Date(as.POSIXct(TILDate,format = '%Y-%m-%d',tz = 'GMT'),tz = 'GMT')) %>%
  group_by(GUPI) %>%
  mutate(dTILTimepoint = c(NA,diff(TILTimepoint)),
         dTILBasic = c(NA,diff(TILBasic)))

# Merge study window and TILBasic information and format days
study.days.prepostTILBasic <- study.windows %>%
  full_join(formatted.TIL.values) %>%
  filter(TILTimepoint %in% TIL.assessment.days,
         TILTimepoint>1,
         !is.na(dTILBasic)) %>%
  rename(postTILBasic = TILBasic) %>%
  mutate(preTILBasic = postTILBasic - dTILBasic,
         ICUDay = sprintf('Day %.0f',TILTimepoint),
         Grouping = case_when(TILTimepoint<=7~'1',
                              TILTimepoint<=10~'2',
                              TILTimepoint<=14~'3',
                              TILTimepoint<=21~'4',
                              TILTimepoint<=28~'5'),
         preTILBasic = factor(preTILBasic),
         postTILBasic = factor(postTILBasic)) %>%
  count(ICUDay, Grouping, preTILBasic, postTILBasic) %>%
  group_by(ICUDay, Grouping, preTILBasic) %>%
  mutate(pct=100*(n/sum(n)),
         # Label = paste0(as.character(signif(pct,2)),'%\n(',n,')')) %>%
         Label = paste0(as.character(signif(pct,2)),'%')) %>%
  rbind(data.frame(ICUDay = 'Day 1',
                   Grouping = '1')) %>%
  mutate(ICUDay=factor(ICUDay,levels=TIL.day.labels),
         TextColor = case_when(postTILBasic==2 ~ 'black',
                               T~'white'))

## Plot and save distribution of next-day TILBasic given previous-day TILBasic
# Create ggplot object of distribution of next-day TILBasic for each previous-day TILBasic
prepost.TILBasic.0 <- study.days.prepostTILBasic %>%
  filter((preTILBasic==0)|(ICUDay=='Day 1')) %>%
  pre.post.TILBasic.dist.plots('Prior-day TIL(Basic) = 0')
prepost.TILBasic.1 <- study.days.prepostTILBasic %>%
  filter((preTILBasic==1)|(ICUDay=='Day 1')) %>%
  pre.post.TILBasic.dist.plots('Prior-day TIL(Basic) = 1')
prepost.TILBasic.2 <- study.days.prepostTILBasic %>%
  filter((preTILBasic==2)|(ICUDay=='Day 1')) %>%
  pre.post.TILBasic.dist.plots('Prior-day TIL(Basic) = 2')
prepost.TILBasic.3 <- study.days.prepostTILBasic %>%
  filter((preTILBasic==3)|(ICUDay=='Day 1')) %>%
  pre.post.TILBasic.dist.plots('Prior-day TIL(Basic) = 3')
prepost.TILBasic.4 <- study.days.prepostTILBasic %>%
  filter((preTILBasic==4)|(ICUDay=='Day 1')) %>%
  pre.post.TILBasic.dist.plots('Prior-day TIL(Basic) = 4')

# Compile ggplot object across previous-day TILBasic values
compiled.prepost.TILBasic <- ggarrange(prepost.TILBasic.0,
                                       prepost.TILBasic.1,
                                       prepost.TILBasic.2,
                                       prepost.TILBasic.3,
                                       prepost.TILBasic.4,
                                       ncol = 2, nrow = 3)

# Create directory for current date and save distribution of next-day TILBasic given previous-day TILBasic
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'prepost_TIL_Basic_distributions_compiled.svg'),compiled.prepost.TILBasic,device=svglite,units='in',dpi=600,width=7.5,height=8.5)

### Visualise the distribution of previous-day and next-day TILBasic given a transition occurred
## Load and prepare dataframes
# Load and clean dataframe containing ICU daily windows of study participants
study.windows <- read.csv('../../center_tbi/CENTER-TBI/FormattedTIL/study_window_timestamps_outcomes.csv',na.strings = c("NA","NaN","", " ")) %>%
  select(GUPI,WindowIdx,WindowTotal,TimeStampStart,TimeStampEnd) %>%
  rename(TILTimepoint = WindowIdx) %>%
  mutate(TimeStampStart = as.POSIXct(TimeStampStart,format = '%Y-%m-%d',tz = 'GMT'),
         TimeStampEnd = as.POSIXct(TimeStampEnd,format = '%Y-%m-%d %H:%M:%S',tz = 'GMT'),
         TILDate = as.Date(TimeStampStart,tz = 'GMT'))

# Load and clean dataframe containing formatted TIL scores
formatted.TIL.values <- read.csv('../../center_tbi/CENTER-TBI/FormattedTIL/formatted_TIL_values.csv',na.strings = c("NA","NaN","", " ")) %>%
  select(GUPI,TILTimepoint,TILDate,TILBasic) %>%
  mutate(TILDate = as.Date(as.POSIXct(TILDate,format = '%Y-%m-%d',tz = 'GMT'),tz = 'GMT')) %>%
  group_by(GUPI) %>%
  mutate(dTILTimepoint = c(NA,diff(TILTimepoint)),
         dTILBasic = c(NA,diff(TILBasic)))

# Merge study window and TILBasic information and format days
study.days.transTILBasic <- study.windows %>%
  full_join(formatted.TIL.values) %>%
  filter(TILTimepoint %in% TIL.assessment.days,
         TILTimepoint>1,
         (!is.na(dTILBasic))&(dTILBasic!=0)) %>%
  rename(postTILBasic = TILBasic) %>%
  mutate(preTILBasic = postTILBasic - dTILBasic,
         preICUDay = sprintf('Day %.0f',TILTimepoint-dTILTimepoint),
         postICUDay = sprintf('Day %.0f',TILTimepoint)) %>%
  pivot_longer(cols=c(preICUDay,postICUDay),names_to ='PreOrPost',values_to ='ICUDay') %>%
  filter(ICUDay %in% TIL.day.labels) %>%
  mutate(Grouping = case_when(ICUDay %in% paste('Day',1:7)~'1',
                              ICUDay %in% paste('Day',10)~'2',
                              ICUDay %in% paste('Day',14)~'3',
                              ICUDay %in% paste('Day',21)~'4',
                              ICUDay %in% paste('Day',28)~'5'),
         TILBasic = case_when(PreOrPost=='preICUDay'~preTILBasic,
                              PreOrPost=='postICUDay'~postTILBasic),
         TILBasic = factor(TILBasic)) %>%
  count(PreOrPost, ICUDay, Grouping, TILBasic) %>%
  group_by(PreOrPost, ICUDay, Grouping) %>%
  mutate(pct=100*(n/sum(n)),
         Label = paste0(as.character(signif(pct,2)),'%\n(',n,')')) %>%
  # Label = paste0(as.character(signif(pct,2)),'%')) %>%
  rbind(data.frame(ICUDay = 'Day 1',
                   Grouping = '1',
                   PreOrPost='postICUDay')) %>%
  mutate(ICUDay=factor(ICUDay,levels=TIL.day.labels),
         TextColor = case_when(TILBasic==2 ~ 'black',
                               T~'white'))

## Plot and save distribution of next-day TILBasic given previous-day TILBasic
# Create ggplot object of previous-day TILBasic given a transition occurred
pre.trans.TILBasic.dist <- study.days.transTILBasic %>%
  filter((PreOrPost=='preICUDay')) %>%
  ggplot(aes(fill=fct_rev(TILBasic), y=pct, x=ICUDay)) + 
  geom_bar(position="stack", stat="identity") +
  geom_text(aes(label = Label),
            position = position_stack(vjust = .5),
            size=6/.pt,
            family='Roboto Condensed',
            color='white') +
  scale_fill_manual(values=rev(c(BluRedDiv5))) +
  # guides(fill=guide_legend(title="TIL(Basic)",nrow = 1,reverse = T)) +
  scale_y_continuous(expand = expansion(mult = c(.00, .00)))+
  scale_color_manual(values = c('black','white'),breaks = c('black','white'),guide='none') +
  theme_minimal(base_family = 'Roboto Condensed') +
  ylab('Percentage (%)') +
  xlab('Day of ICU stay') +
  ggtitle('Distribution of TIL(Basic) directly preceding transition')+
  facet_grid(cols = vars(Grouping), scales = 'free_x', switch = 'x', space = 'free_x') +
  theme(
    plot.title = element_text(size=8, color = "black",face = 'bold',margin = margin(b = .5),hjust = .5),
    strip.text = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_blank(),
    panel.border = element_blank(),
    panel.spacing = unit(5, 'points'),
    axis.text.x = element_text(size = 6, color = "black",margin = margin(0,0,0,0)),
    axis.text.y = element_text(size = 6, color = "black",margin = margin(0,0,0,0)),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold',margin = margin(0,0,0,0)),
    legend.position = 'none'
  )

# Create ggplot object of next-day TILBasic given a transition occurred
post.trans.TILBasic.dist <- study.days.transTILBasic %>%
  filter((PreOrPost=='postICUDay')) %>%
  ggplot(aes(fill=fct_rev(TILBasic), y=pct, x=ICUDay)) + 
  geom_bar(position="stack", stat="identity") +
  geom_text(aes(label = Label),
            position = position_stack(vjust = .5),
            size=6/.pt,
            family='Roboto Condensed',
            color='white') +
  scale_fill_manual(values=rev(c(BluRedDiv5))) +
  # guides(fill=guide_legend(title="TIL(Basic)",nrow = 1,reverse = T)) +
  scale_y_continuous(expand = expansion(mult = c(.00, .00)))+
  scale_color_manual(values = c('black','white'),breaks = c('black','white'),guide='none') +
  theme_minimal(base_family = 'Roboto Condensed') +
  ylab('Percentage (%)') +
  xlab('Day of ICU stay') +
  ggtitle('Distribution of TIL(Basic) directly following transition')+
  facet_grid(cols = vars(Grouping), scales = 'free_x', switch = 'x', space = 'free_x') +
  theme(
    plot.title = element_text(size=8, color = "black",face = 'bold',margin = margin(b = .5),hjust = .5),
    strip.text = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_blank(),
    panel.border = element_blank(),
    panel.spacing = unit(5, 'points'),
    axis.text.x = element_text(size = 6, color = "black",margin = margin(0,0,0,0)),
    axis.text.y = element_text(size = 6, color = "black",margin = margin(0,0,0,0)),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold',margin = margin(0,0,0,0)),
    legend.position = 'none'
  )

# Compile ggplot object of pre- and post-transition distributions
compiled.prepost.transTILBasic <- ggarrange(pre.trans.TILBasic.dist,
                                            post.trans.TILBasic.dist,
                                            ncol = 2, nrow = 1)

# Create directory for current date and save distribution of next-day TILBasic given previous-day TILBasic
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'trans_prepost_TIL_Basic_distributions.svg'),compiled.prepost.transTILBasic,device=svglite,units='in',dpi=600,width=7.5,height=(8.5/3))

### Visualise feature TimeSHAP values across all points of TILBasic transition 
## Load and prepare dataframes
# Load and clean dataframe containing feature TimeSHAP values
filt.timeSHAP.df <- read.csv('../TILTomorrow_model_interpretations/v2-0/timeSHAP/viz_feature_timeSHAP_values.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter(VIZ_IDX %in% c(133,293)) %>%
  mutate(VIZ_IDX = factor(VIZ_IDX),
         GROUPS = case_when((PlotIdx >= 11) ~ 'Top',
                            (PlotIdx <= 10) ~'Bottom'),
         GROUPS = factor(GROUPS,levels=c('Top','Middle','Bottom')))

## Plot and save TimeSHAP plots of two different models at points of transition
# Create ggplot object for feature importance beeswarm plot for full model
full.model.timeSHAP.plot <- filt.timeSHAP.df %>%
  filter(VIZ_IDX==293) %>%
  mutate(Label = fct_reorder(Label, PlotIdx)) %>%
  ggplot() +
  coord_cartesian(xlim = c(-1,1)) +
  scale_color_gradient2(na.value='#488f31',low='#003f5c',mid='#eacaf4',high='#de425b',midpoint=.5,limits = c(0,1),breaks = c(0.05,.95), labels = c('Low','High')) +
  geom_vline(xintercept = 0, color = "darkgray") +
  geom_quasirandom(aes(y=Label,x=METRIC,color=ColorScale),groupOnX=FALSE,varwidth=FALSE,alpha = .8,stroke = 0,size=1) + 
  theme_minimal(base_family = 'Roboto Condensed') +
  guides(color = guide_colourbar(title='Feature Value',title.vjust=1,barwidth = 10, barheight = .25,ticks = FALSE))+
  facet_grid(rows = vars(GROUPS), scales = 'free_y', switch = 'y', space = 'free_y') +
  theme(
    strip.background = element_blank(),
    strip.text = element_blank(),
    axis.title.y = element_blank(),
    axis.text.x = element_text(size = 5, color = 'black'),
    axis.text.y = element_blank(),
    axis.title.x = element_blank(),
    panel.border = element_blank(),
    axis.line.x = element_line(size=1/.pt),
    axis.text = element_text(color='black'),
    legend.position = 'none',
    panel.grid.major.y = element_blank(),
    panel.spacing = unit(10, 'points'),
    plot.margin=grid::unit(c(0,2,0,0), "mm")
  )

# Create ggplot object for feature importance beeswarm plot for full model
no.clinician.model.timeSHAP.plot <- filt.timeSHAP.df %>%
  filter(VIZ_IDX==133,
  ) %>%
  mutate(Label = fct_reorder(Label, PlotIdx)) %>%
  ggplot() +
  coord_cartesian(xlim = c(-.75,.75)) +
  scale_color_gradient2(na.value='#488f31',low='#003f5c',mid='#eacaf4',high='#de425b',midpoint=.5,limits = c(0,1),breaks = c(0.05,.95), labels = c('Low','High')) +
  geom_vline(xintercept = 0, color = "darkgray") +
  geom_quasirandom(aes(y=Label,x=METRIC,color=ColorScale),groupOnX=FALSE,varwidth=FALSE,alpha = .8,stroke = 0,size=1) + 
  theme_minimal(base_family = 'Roboto Condensed') +
  guides(color = guide_colourbar(title='Feature Value',title.vjust=1,barwidth = 10, barheight = .25,ticks = FALSE))+
  facet_grid(rows = vars(GROUPS), scales = 'free_y', switch = 'y', space = 'free_y') +
  theme(
    strip.background = element_blank(),
    strip.text = element_blank(),
    axis.title.y = element_blank(),
    axis.text.x = element_text(size = 5, color = 'black'),
    axis.text.y = element_blank(),
    # axis.text.y = element_text(color = 'black',angle = 30, hjust=1),
    axis.title.x = element_blank(),
    panel.border = element_blank(),
    axis.line.x = element_line(size=1/.pt),
    axis.text = element_text(color='black'),
    legend.position = 'none',
    panel.grid.major.y = element_blank(),
    panel.spacing = unit(10, 'points'),
    plot.margin=grid::unit(c(0,2,0,0), "mm")
  )

# Create directory for current date and save feature-level TimeSHAP plots
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'full_model_timeshap.png'),full.model.timeSHAP.plot,units='in',dpi=600,height=3.75,width=2.47)
ggsave(file.path('../plots',Sys.Date(),'no_clinician_timeshap.png'),no.clinician.model.timeSHAP.plot,units='in',dpi=600,height=3.75,width=2.47)

### Visualise feature TimeSHAP at all points of TILBasic transition for each starting TILBasic value
## Load and prepare dataframes
# Load and clean dataframe containing feature TimeSHAP values
filt.timeSHAP.df <- read.csv('../TILTomorrow_model_interpretations/v2-0/timeSHAP/viz_feature_timeSHAP_values.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter(VIZ_IDX %in% c(132,292)) %>%
  mutate(VIZ_IDX = case_when((VIZ_IDX == 132) ~ 'Models without clinician impressions or treatments',
                             (VIZ_IDX == 292) ~'Models with full variable set'),
         VIZ_IDX = factor(VIZ_IDX,levels=c('Models with full variable set',
                                           'Models without clinician impressions or treatments')),
         GROUPS = case_when((TILBasic == 0) ~ 'Bottom',
                            (TILBasic == 4) ~ 'Top',
                            (PlotIdx >= 11) ~ 'Top',
                            (PlotIdx <= 10) ~'Bottom'),
         GROUPS = factor(GROUPS,levels=c('Top','Middle','Bottom')),
         TILBasic = factor(paste('TIL(Basic) =',TILBasic)))

## Plot and save TimeSHAP plots of two different models at points of transition for each starting TILBasic value
# Create ggplot object for full model TimeSHAP with TILBasic = 0
TILBasic.0.full.plot <- filt.timeSHAP.df %>%
  filter(VIZ_IDX=='Models with full variable set',
         TILBasic=='TIL(Basic) = 0') %>%
  mutate(Label = fct_reorder(Label, PlotIdx)) %>%
  TILBasic.timeSHAP.plots()

# Create ggplot object for limited model TimeSHAP with TILBasic = 0
TILBasic.0.limited.plot <- filt.timeSHAP.df %>%
  filter(VIZ_IDX=='Models without clinician impressions or treatments',
         TILBasic=='TIL(Basic) = 0') %>%
  mutate(Label = fct_reorder(Label, PlotIdx)) %>%
  TILBasic.timeSHAP.plots()

# Create ggplot object for full model TimeSHAP with TILBasic = 1
TILBasic.1.full.plot <- filt.timeSHAP.df %>%
  filter(VIZ_IDX=='Models with full variable set',
         TILBasic=='TIL(Basic) = 1') %>%
  mutate(Label = fct_reorder(Label, PlotIdx)) %>%
  TILBasic.timeSHAP.plots()

# Create ggplot object for limited model TimeSHAP with TILBasic = 1
TILBasic.1.limited.plot <- filt.timeSHAP.df %>%
  filter(VIZ_IDX=='Models without clinician impressions or treatments',
         TILBasic=='TIL(Basic) = 1') %>%
  mutate(Label = fct_reorder(Label, PlotIdx)) %>%
  TILBasic.timeSHAP.plots()

# Create ggplot object for full model TimeSHAP with TILBasic = 2
TILBasic.2.full.plot <- filt.timeSHAP.df %>%
  filter(VIZ_IDX=='Models with full variable set',
         TILBasic=='TIL(Basic) = 2') %>%
  mutate(Label = fct_reorder(Label, PlotIdx)) %>%
  TILBasic.timeSHAP.plots()

# Create ggplot object for limited model TimeSHAP with TILBasic = 2
TILBasic.2.limited.plot <- filt.timeSHAP.df %>%
  filter(VIZ_IDX=='Models without clinician impressions or treatments',
         TILBasic=='TIL(Basic) = 2') %>%
  mutate(Label = fct_reorder(Label, PlotIdx)) %>%
  TILBasic.timeSHAP.plots()

# Create ggplot object for full model TimeSHAP with TILBasic = 3
TILBasic.3.full.plot <- filt.timeSHAP.df %>%
  filter(VIZ_IDX=='Models with full variable set',
         TILBasic=='TIL(Basic) = 3') %>%
  mutate(Label = fct_reorder(Label, PlotIdx)) %>%
  TILBasic.timeSHAP.plots()

# Create ggplot object for limited model TimeSHAP with TILBasic = 3
TILBasic.3.limited.plot <- filt.timeSHAP.df %>%
  filter(VIZ_IDX=='Models without clinician impressions or treatments',
         TILBasic=='TIL(Basic) = 3') %>%
  mutate(Label = fct_reorder(Label, PlotIdx)) %>%
  TILBasic.timeSHAP.plots()

# Create ggplot object for full model TimeSHAP with TILBasic = 4
TILBasic.4.full.plot <- filt.timeSHAP.df %>%
  filter(VIZ_IDX=='Models with full variable set',
         TILBasic=='TIL(Basic) = 4') %>%
  mutate(Label = fct_reorder(Label, PlotIdx)) %>%
  TILBasic.timeSHAP.plots()

# Create ggplot object for limited model TimeSHAP with TILBasic = 4
TILBasic.4.limited.plot <- filt.timeSHAP.df %>%
  filter(VIZ_IDX=='Models without clinician impressions or treatments',
         TILBasic=='TIL(Basic) = 4') %>%
  mutate(Label = fct_reorder(Label, PlotIdx)) %>%
  TILBasic.timeSHAP.plots()

# Create directory for current date and save TILBasic-stratified feature-level TimeSHAP plots
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'TILBasic_0_full_model_timeshap.png'),TILBasic.0.full.plot,units='in',dpi=600,height=3.38,width=3.75)
ggsave(file.path('../plots',Sys.Date(),'TILBasic_0_limited_model_timeshap.png'),TILBasic.0.limited.plot,units='in',dpi=600,height=3.38,width=3.75)
ggsave(file.path('../plots',Sys.Date(),'TILBasic_1_full_model_timeshap.png'),TILBasic.1.full.plot,units='in',dpi=600,height=3.38,width=3.75)
ggsave(file.path('../plots',Sys.Date(),'TILBasic_1_limited_model_timeshap.png'),TILBasic.1.limited.plot,units='in',dpi=600,height=3.38,width=3.75)
ggsave(file.path('../plots',Sys.Date(),'TILBasic_2_full_model_timeshap.png'),TILBasic.2.full.plot,units='in',dpi=600,height=3.38,width=3.75)
ggsave(file.path('../plots',Sys.Date(),'TILBasic_2_limited_model_timeshap.png'),TILBasic.2.limited.plot,units='in',dpi=600,height=3.38,width=3.75)
ggsave(file.path('../plots',Sys.Date(),'TILBasic_3_full_model_timeshap.png'),TILBasic.3.full.plot,units='in',dpi=600,height=3.38,width=3.75)
ggsave(file.path('../plots',Sys.Date(),'TILBasic_3_limited_model_timeshap.png'),TILBasic.3.limited.plot,units='in',dpi=600,height=3.38,width=3.75)
ggsave(file.path('../plots',Sys.Date(),'TILBasic_4_full_model_timeshap.png'),TILBasic.4.full.plot,units='in',dpi=600,height=3.38,width=3.75)
ggsave(file.path('../plots',Sys.Date(),'TILBasic_4_limited_model_timeshap.png'),TILBasic.4.limited.plot,units='in',dpi=600,height=3.38,width=3.75)

### Visualise missing feature TimeSHAP at all points of TILBasic transition
## Load and prepare dataframes
# Load and clean dataframe containing feature TimeSHAP values
filt.timeSHAP.df <- read.csv('../TILTomorrow_model_interpretations/v2-0/timeSHAP/viz_feature_timeSHAP_values.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter(VIZ_IDX %in% c(131,291)) %>%
  mutate(VIZ_IDX = case_when((VIZ_IDX == 131) ~ 'Models without clinician impressions or treatments',
                             (VIZ_IDX == 291) ~'Models with full variable set'),
         VIZ_IDX = factor(VIZ_IDX,levels=c('Models with full variable set',
                                           'Models without clinician impressions or treatments')),
         GROUPS = case_when((PlotIdx >= 11) ~ 'Top',
                            (PlotIdx <= 10) ~'Bottom'),
         GROUPS = factor(GROUPS,levels=c('Top','Middle','Bottom')))

## Plot and save missing value TimeSHAP plots of two different models at points of transition
# Create ggplot object for full model TimeSHAP of missing values
missing.val.full.plot <- filt.timeSHAP.df %>%
  filter(VIZ_IDX=='Models with full variable set') %>%
  mutate(Label = fct_reorder(Label, PlotIdx)) %>%
  TILBasic.timeSHAP.plots()

# Create ggplot object for full model TimeSHAP of missing values
missing.val.limited.plot <- filt.timeSHAP.df %>%
  filter(VIZ_IDX=='Models without clinician impressions or treatments') %>%
  mutate(Label = fct_reorder(Label, PlotIdx)) %>%
  TILBasic.timeSHAP.plots()

# Create directory for current date and save TILBasic-stratified feature-level TimeSHAP plots
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'missing_value_full_model_timeshap.png'),missing.val.full.plot,units='in',dpi=600,height=3.38,width=3.75)
ggsave(file.path('../plots',Sys.Date(),'missing_value_limited_model_timeshap.png'),missing.val.limited.plot,units='in',dpi=600,height=3.38,width=3.75)

### Visualise threshold-level AUC in prediction of next-day TILBasic
## Load and prepare dataframes
# Load and clean dataframe containing performance metrics of full model
full.model.AUC.CIs <- read.csv('../TILTomorrow_model_performance/v2-0/test_set_metrics_CI.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter(METRIC=='AUC',
         TUNE_IDX==332) %>%
  mutate(ICUDay = sprintf('%.0f',WINDOW_IDX),
         Grouping = case_when(WINDOW_IDX<=6~'1',
                              WINDOW_IDX<=9~'2',
                              WINDOW_IDX<=13~'3'),
         ICUDay=fct_reorder(factor(ICUDay), WINDOW_IDX),
         VariableSet = 'Full')

# Load and clean dataframe containing performance metrics of models trained without dynamic variables and without clinician impressions/treatments 
limited.model.AUC.CIs <- read.csv('../TILTomorrow_model_performance/v2-0/sens_analysis_metrics_CI.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter(METRIC=='AUC',
         TUNE_IDX==332,
         SENS_IDX %in% c(1,4)) %>%
  mutate(ICUDay = sprintf('%.0f',WINDOW_IDX),
         Grouping = case_when(WINDOW_IDX<=6~'1',
                              WINDOW_IDX<=9~'2',
                              WINDOW_IDX<=13~'3'),
         ICUDay=fct_reorder(factor(ICUDay), WINDOW_IDX),
         VariableSet = case_when(SENS_IDX==1~'No dynamic',
                                 SENS_IDX==4~'No clinician impressions or treatments'))

# Load and clean dataframe containing performance metrics of last-TILBasic-carried forward 
no.info.AUC.CIs <- read.csv('../TILTomorrow_model_performance/v2-0/no_information_metrics_CI.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter(METRIC=='AUC',
         TUNE_IDX==332) %>%
  mutate(ICUDay = sprintf('%.0f',WINDOW_IDX),
         Grouping = case_when(WINDOW_IDX<=6~'1',
                              WINDOW_IDX<=9~'2',
                              WINDOW_IDX<=13~'3'),
         ICUDay=fct_reorder(factor(ICUDay), WINDOW_IDX),
         VariableSet = 'Only last available TIL(Basic)')

# Combine all-point AUC values into single dataframe
AUC.plot.df <- full.model.AUC.CIs %>%
  filter(METRIC=='AUC',
         TUNE_IDX==332) %>%
  select(TUNE_IDX,ICUDay,THRESHOLD,lo,median,hi,Grouping,VariableSet) %>%
  rbind(limited.model.AUC.CIs %>%
          filter(METRIC=='AUC',
                 TUNE_IDX==332) %>%
          select(TUNE_IDX,ICUDay,THRESHOLD,lo,median,hi,Grouping,VariableSet)) %>%
  rbind(no.info.AUC.CIs %>%
          select(TUNE_IDX,THRESHOLD,ICUDay,lo,median,hi,Grouping,VariableSet)) %>%
  mutate(VariableSet = factor(VariableSet,levels=c('Full','Only last available TIL(Basic)','No clinician impressions or treatments','No dynamic')))

# Load and clean dataframe containing performance metrics of full model at points of transition
trans.full.model.AUC.CIs <- read.csv('../TILTomorrow_model_performance/v2-0/trans_test_set_metrics_CI.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter(METRIC=='AUC',
         TUNE_IDX==332) %>%
  mutate(ICUDay = sprintf('%.0f',WINDOW_IDX),
         Grouping = case_when(WINDOW_IDX<=6~'1',
                              WINDOW_IDX<=9~'2',
                              WINDOW_IDX<=13~'3'),
         ICUDay=fct_reorder(factor(ICUDay), WINDOW_IDX),
         VariableSet = 'Full')

# Load and clean dataframe containing performance metrics of models trained without dynamic variables and without clinician impressions/treatments at points of transition
trans.limited.model.AUC.CIs <- read.csv('../TILTomorrow_model_performance/v2-0/trans_sens_analysis_metrics_CI.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter(METRIC=='AUC',
         TUNE_IDX==332,
         SENS_IDX %in% c(1,4)) %>%
  mutate(ICUDay = sprintf('%.0f',WINDOW_IDX),
         Grouping = case_when(WINDOW_IDX<=6~'1',
                              WINDOW_IDX<=9~'2',
                              WINDOW_IDX<=13~'3'),
         ICUDay=fct_reorder(factor(ICUDay), WINDOW_IDX),
         VariableSet = case_when(SENS_IDX==1~'No dynamic',
                                 SENS_IDX==4~'No clinician impressions or treatments'))

# Load and clean dataframe containing performance metrics of last-TILBasic carried forward at points of transition
trans.no.info.AUC.CIs <- read.csv('../TILTomorrow_model_performance/v2-0/trans_no_information_metrics_CI.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter(METRIC=='AUC',
         TUNE_IDX==332) %>%
  mutate(ICUDay = sprintf('%.0f',WINDOW_IDX),
         Grouping = case_when(WINDOW_IDX<=6~'1',
                              WINDOW_IDX<=9~'2',
                              WINDOW_IDX<=13~'3'),
         ICUDay=fct_reorder(factor(ICUDay), WINDOW_IDX),
         VariableSet = 'Only last available TILBasic')

# Combine transition-point AUC values into single dataframe
trans.AUC.plot.df <- trans.full.model.AUC.CIs %>%
  filter(METRIC=='AUC',
         TUNE_IDX==332) %>%
  select(TUNE_IDX,ICUDay,THRESHOLD,lo,median,hi,Grouping,VariableSet) %>%
  rbind(trans.limited.model.AUC.CIs %>%
          filter(METRIC=='AUC',
                 TUNE_IDX==332) %>%
          select(TUNE_IDX,ICUDay,THRESHOLD,lo,median,hi,Grouping,VariableSet)) %>%
  rbind(trans.no.info.AUC.CIs %>%
          select(TUNE_IDX,THRESHOLD,ICUDay,lo,median,hi,Grouping,VariableSet)) %>%
  mutate(VariableSet = factor(VariableSet,levels=c('Full','Only last available TILBasic','No clinician impressions or treatments','No dynamic')))

## Plot and save next-day prediction AUC plots
# All-point TILBasic>0 AUCs
all.point.TILBasic.0.AUC <- AUC.plot.df %>%
  filter(THRESHOLD == 'TILBasic>0') %>%
  thresh.level.AUC.plot('Next-day TIL(Basic) > 0',Palette4)

# All-point TILBasic>1 AUCs
all.point.TILBasic.1.AUC <- AUC.plot.df %>%
  filter(THRESHOLD == 'TILBasic>1') %>%
  thresh.level.AUC.plot('Next-day TIL(Basic) > 1',Palette4)

# All-point TILBasic>2 AUCs
all.point.TILBasic.2.AUC <- AUC.plot.df %>%
  filter(THRESHOLD == 'TILBasic>2') %>%
  thresh.level.AUC.plot('Next-day TIL(Basic) > 2',Palette4)

# All-point TILBasic>3 AUCs
all.point.TILBasic.3.AUC <- AUC.plot.df %>%
  filter(THRESHOLD == 'TILBasic>3') %>%
  thresh.level.AUC.plot('Next-day TIL(Basic) > 3',Palette4)

# Compile ggplot objects of all-point, threshold-level AUC plots
all.point.thresh.AUCs <- ggarrange(all.point.TILBasic.0.AUC,
                                   all.point.TILBasic.1.AUC,
                                   all.point.TILBasic.2.AUC,
                                   all.point.TILBasic.3.AUC,
                                   ncol = 4, nrow = 1)

# Transition-point TILBasic>0 AUCs
trans.point.TILBasic.0.AUC <- trans.AUC.plot.df %>%
  filter(THRESHOLD == 'TILBasic>0') %>%
  thresh.level.AUC.plot('Next-day TIL(Basic) > 0',Palette4)

# Transition-point TILBasic>1 AUCs
trans.point.TILBasic.1.AUC <- trans.AUC.plot.df %>%
  filter(THRESHOLD == 'TILBasic>1') %>%
  thresh.level.AUC.plot('Next-day TIL(Basic) > 1',Palette4)

# Transition-point TILBasic>2 AUCs
trans.point.TILBasic.2.AUC <- trans.AUC.plot.df %>%
  filter(THRESHOLD == 'TILBasic>2') %>%
  thresh.level.AUC.plot('Next-day TIL(Basic) > 2',Palette4)

# Transition-point TILBasic>3 AUCs
trans.point.TILBasic.3.AUC <- trans.AUC.plot.df %>%
  filter(THRESHOLD == 'TILBasic>3') %>%
  thresh.level.AUC.plot('Next-day TIL(Basic) > 3',Palette4)

# Compile ggplot objects of transition-point, threshold-level AUC plots
trans.point.thresh.AUCs <- ggarrange(trans.point.TILBasic.0.AUC,
                                     trans.point.TILBasic.1.AUC,
                                     trans.point.TILBasic.2.AUC,
                                     trans.point.TILBasic.3.AUC,
                                     ncol = 4, nrow = 1)

# Create directory for current date and save threshold-level AUC plots for next-day TILBasic prediction
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'all_point_thresh_AUCs.svg'),all.point.thresh.AUCs,device= svglite,units='in',dpi=600,width=7.405,height = 1.875)
ggsave(file.path('../plots',Sys.Date(),'trans_point_thresh_AUCs.svg'),trans.point.thresh.AUCs,device= svglite,units='in',dpi=600,width=7.405,height = 1.875)

### Visualise threshold-level calibration slope in prediction of next-day TILBasic
## Load and prepare dataframes
# Load and clean dataframe containing performance metrics of full model
full.model.calib.slope.CIs <- read.csv('../TILTomorrow_model_performance/v2-0/test_set_metrics_CI.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter(METRIC=='CALIB_SLOPE',
         TUNE_IDX==332) %>%
  mutate(ICUDay = sprintf('%.0f',WINDOW_IDX),
         Grouping = case_when(WINDOW_IDX<=6~'1',
                              WINDOW_IDX<=9~'2',
                              WINDOW_IDX<=13~'3'),
         ICUDay=fct_reorder(factor(ICUDay), WINDOW_IDX),
         VariableSet = 'Full')

## Plot and save next-day prediction calibration slope plots
# TILBasic>0 calibration plots
all.point.TILBasic.0.calib.slope <- full.model.calib.slope.CIs %>%
  filter(THRESHOLD == 'TILBasic>0') %>%
  thresh.level.calib.slope.plot('Next-day TIL(Basic) > 0')

# TILBasic>1 calibration plots
all.point.TILBasic.1.calib.slope <- full.model.calib.slope.CIs %>%
  filter(THRESHOLD == 'TILBasic>1') %>%
  thresh.level.calib.slope.plot('Next-day TIL(Basic) > 1')

# TILBasic>2 calibration plots
all.point.TILBasic.2.calib.slope <- full.model.calib.slope.CIs %>%
  filter(THRESHOLD == 'TILBasic>2') %>%
  thresh.level.calib.slope.plot('Next-day TIL(Basic) > 2')

# TILBasic>3 calibration plots
all.point.TILBasic.3.calib.slope <- full.model.calib.slope.CIs %>%
  filter(THRESHOLD == 'TILBasic>3') %>%
  thresh.level.calib.slope.plot('Next-day TIL(Basic) > 3')

# Compile ggplot objects of all-point, threshold-level calibration slope plots
all.point.thresh.calib.slope <- ggarrange(all.point.TILBasic.0.calib.slope,
                                          all.point.TILBasic.1.calib.slope,
                                          all.point.TILBasic.2.calib.slope,
                                          all.point.TILBasic.3.calib.slope,
                                          ncol = 2, nrow = 2)

# Create directory for current date and save threshold-level AUC plots for next-day TILBasic prediction
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'all_point_thresh_calib_slopes.svg'),all.point.thresh.calib.slope,device= svglite,units='in',dpi=600,width=3.75,height = 2.75)

### Visualise threshold-level calibration curves in prediction of next-day TILBasic
## Load and prepare dataframes
# Load and clean dataframe containing performance metrics of full model
full.model.calib.curve.CIs <- read.csv('../TILTomorrow_model_performance/v2-0/test_set_calibration_curves_CI.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter(TUNE_IDX==332) %>%
  mutate(ICUDay = sprintf('%.0f',WINDOW_IDX),
         Grouping = case_when(WINDOW_IDX<=6~'1',
                              WINDOW_IDX<=9~'2',
                              WINDOW_IDX<=13~'3'),
         lo = case_when(lo<0~0,
                        lo>1~1,
                        T~lo),
         median = case_when(median<0~0,
                            median>1~1,
                            T~median),
         hi = case_when(hi<0~0,
                        hi>1~1,
                        T~hi),
         ICUDay=fct_reorder(factor(ICUDay), WINDOW_IDX),
         VariableSet = 'Full',
         THRESHOLD = str_replace(THRESHOLD,'TILBasic>','Next-day TIL(Basic) > '))

## Plot and save next-day prediction calibration curve plots
# TILBasic calibration curve
all.point.TILBasic.calib.curve <- full.model.calib.curve.CIs %>%
  filter(ICUDay %in% c('1','2','6','13')) %>%
  ggplot(aes(x=100*PREDPROB)) +
  facet_wrap(~THRESHOLD, scales = 'free',ncol = 3) +
  coord_cartesian(ylim = c(0,100),xlim = c(0,100))+
  geom_segment(x = 0, y = 0, xend = 100, yend = 100,alpha = 0.5,linetype = "dashed",size=.75/.pt, color = 'gray')+
  geom_ribbon(aes(ymin = 100*lo, ymax = 100*hi, fill = ICUDay), alpha = 0.3,size=.75/.pt,color=NA) +
  geom_line(aes(y = 100*median, color = ICUDay), alpha = 1, size=1.3/.pt) +
  scale_x_continuous(expand = expansion(mult = c(.01, .01))) +
  scale_y_continuous(expand = expansion(mult = c(.01, .01))) +
  guides(fill=guide_legend(nrow=1,byrow=TRUE),color=guide_legend(nrow=1,byrow=TRUE)) +
  scale_fill_manual(name = "Day of ICU stay",values = Palette4)+
  scale_color_manual(name = "Day of ICU stay",values = Palette4)+
  xlab("Predicted probability (%)") +
  ylab("Observed probability (%)") +
  theme_classic(base_family = 'Roboto Condensed') +
  theme(
    strip.text = element_text(size=7, color = "black",face = 'bold',margin = margin(b = .5)), 
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    panel.spacing = unit(5, 'points'),
    axis.text.x = element_text(size = 5, color = "black",margin = margin(r = 0)),
    axis.text.y = element_text(size = 5, color = "black",margin = margin(r = 0)),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold'),
    strip.background = element_blank(),
    aspect.ratio = 1,
    panel.border = element_rect(colour = 'black', fill=NA, size = 1/.pt),
    plot.margin=grid::unit(c(0,2,0,0), "mm"),
    legend.position = 'bottom',
    legend.title = element_text(size = 7, color = "black", face = 'bold'),
    legend.text=element_text(size=6),
    axis.line = element_blank(),
    legend.key.size = unit(1.3/.pt,"line")
  )

# Create directory for current date and save threshold-level AUC plots for next-day TILBasic prediction
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'all_point_thresh_calib_curves.svg'),all.point.TILBasic.calib.curve,device= svglite,units='in',dpi=600,width=3.75,height = 3)























### Explanation of ordinal variance in next-day TILBasic
## Load and prepare dataframes for explanation at all points
# Load full-model Somers D at all points
TIL.Basic.Somers.D.CIs <- read.csv('../TILTomorrow_model_performance/v2-0/test_set_metrics_CI.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter(METRIC=='Somers D',
         TUNE_IDX==332) %>%
  mutate(ICUDay = sprintf('Day %.0f',WINDOW_IDX),
         Grouping = case_when(WINDOW_IDX<=6~'1',
                              WINDOW_IDX<=9~'2',
                              WINDOW_IDX<=13~'3'),
         ICUDay=fct_reorder(factor(ICUDay), WINDOW_IDX),
         VariableSet = 'Full')

# Load sensitivity-analysis Somers D at all points
sens.analysis.Somers.D.CIs <- read.csv('../TILTomorrow_model_performance/v2-0/sens_analysis_metrics_CI.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter(METRIC=='Somers D',
         TUNE_IDX==332,
         SENS_IDX %in% c(1,4)) %>%
  mutate(ICUDay = sprintf('Day %.0f',WINDOW_IDX),
         Grouping = case_when(WINDOW_IDX<=6~'1',
                              WINDOW_IDX<=9~'2',
                              WINDOW_IDX<=13~'3'),
         ICUDay=fct_reorder(factor(ICUDay), WINDOW_IDX),
         VariableSet = case_when(SENS_IDX==1~'No dynamic',
                                 SENS_IDX==4~'No clinician impressions or treatments'))

# Load no-information Somers D at all points
no.info.Somers.D.CIs <- read.csv('../TILTomorrow_model_performance/v2-0/no_information_metrics_CI.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter(METRIC=='Somers D',
         TUNE_IDX==332) %>%
  mutate(ICUDay = sprintf('Day %.0f',WINDOW_IDX),
         Grouping = case_when(WINDOW_IDX<=6~'1',
                              WINDOW_IDX<=9~'2',
                              WINDOW_IDX<=13~'3'),
         ICUDay=fct_reorder(factor(ICUDay), WINDOW_IDX),
         VariableSet = 'Only last available TILBasic')

# Combine all-point Somers D values into single dataframe
Somers.D.plot.df <- TIL.Basic.Somers.D.CIs %>%
  select(TUNE_IDX,ICUDay,lo,median,hi,Grouping,VariableSet) %>%
  rbind(sens.analysis.Somers.D.CIs %>%
          select(TUNE_IDX,ICUDay,lo,median,hi,Grouping,VariableSet)) %>%
  rbind(no.info.Somers.D.CIs %>%
          select(TUNE_IDX,ICUDay,lo,median,hi,Grouping,VariableSet)) %>%
  mutate(VariableSet = factor(VariableSet,levels=c('Full','Only last available TILBasic','No clinician impressions or treatments','No dynamic'))) %>%
  mutate(across(lo:hi, ~ (abs(.x)+.x)/2))

## Load and prepare dataframes for explanation at point of transition
# Load full-model Somers D at point of transition
trans.TIL.Basic.Somers.D.CIs <- read.csv('../TILTomorrow_model_performance/v2-0/trans_test_set_metrics_CI.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter(METRIC=='Somers D',
         TUNE_IDX==332) %>%
  mutate(ICUDay = sprintf('Day %.0f',WINDOW_IDX),
         Grouping = case_when(WINDOW_IDX<=6~'1',
                              WINDOW_IDX<=9~'2',
                              WINDOW_IDX<=13~'3'),
         ICUDay=fct_reorder(factor(ICUDay), WINDOW_IDX),
         VariableSet = 'Full')

# Load sensitivity-analysis Somers D at point of transition
trans.sens.analysis.Somers.D.CIs <- read.csv('../TILTomorrow_model_performance/v2-0/trans_sens_analysis_metrics_CI.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter(METRIC=='Somers D',
         TUNE_IDX==332,
         SENS_IDX %in% c(1,4)) %>%
  mutate(ICUDay = sprintf('Day %.0f',WINDOW_IDX),
         Grouping = case_when(WINDOW_IDX<=6~'1',
                              WINDOW_IDX<=9~'2',
                              WINDOW_IDX<=13~'3'),
         ICUDay=fct_reorder(factor(ICUDay), WINDOW_IDX),
         VariableSet = case_when(SENS_IDX==1~'No dynamic',
                                 SENS_IDX==4~'No clinician impressions or treatments'))

# Load no-information Somers D at point of transition
trans.no.info.Somers.D.CIs <- read.csv('../TILTomorrow_model_performance/v2-0/trans_no_information_metrics_CI.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter(METRIC=='Somers D',
         TUNE_IDX==332) %>%
  mutate(ICUDay = sprintf('Day %.0f',WINDOW_IDX),
         Grouping = case_when(WINDOW_IDX<=6~'1',
                              WINDOW_IDX<=9~'2',
                              WINDOW_IDX<=13~'3'),
         ICUDay=fct_reorder(factor(ICUDay), WINDOW_IDX),
         VariableSet = 'Only last available TILBasic')

# Combine transition-point Somers D values into single dataframe
trans.Somers.D.plot.df <- trans.TIL.Basic.Somers.D.CIs %>%
  select(TUNE_IDX,ICUDay,lo,median,hi,Grouping,VariableSet) %>%
  rbind(trans.sens.analysis.Somers.D.CIs %>%
          select(TUNE_IDX,ICUDay,lo,median,hi,Grouping,VariableSet)) %>%
  rbind(trans.no.info.Somers.D.CIs %>%
          select(TUNE_IDX,ICUDay,lo,median,hi,Grouping,VariableSet)) %>%
  mutate(VariableSet = factor(VariableSet,levels=c('Full','Only last available TILBasic','No clinician impressions or treatments','No dynamic'))) %>%
  mutate(across(lo:hi, ~ (abs(.x)+.x)/2))

## Plot and save TILBasic explanation over days of ICU stay at all points
# Create ggplot object
all.point.Somers.D.plot <- Somers.D.plot.df %>% ggplot() +
  geom_line(data=Somers.D.plot.df %>% filter(Grouping==1),
            mapping=aes(x=ICUDay, y=100*median, color=VariableSet, group = VariableSet),
            lwd=1.3/.pt) +
  geom_ribbon(data=Somers.D.plot.df %>% filter(Grouping==1),
              mapping=aes(x=ICUDay, ymin=100*lo, ymax=100*hi, fill=VariableSet, group = VariableSet),
              alpha=.2) +
  geom_point(data=Somers.D.plot.df %>% filter(Grouping!=1),
             mapping=aes(x=ICUDay, y=100*median, color=VariableSet),
             position = position_dodge(width = .75),
             size=1) +
  geom_errorbar(data=Somers.D.plot.df %>% filter(Grouping!=1),
                mapping=aes(x=ICUDay, ymin=100*lo, ymax=100*hi, color=VariableSet),
                position = position_dodge(width = .75),
                width=.35) +
  coord_cartesian(ylim = c(0,100)) +
  xlab("Day of ICU stay")+
  ylab("Explanation of ordinal variance in next-day TILBasic (%)")+
  scale_y_continuous(breaks = seq(0,100,10)) +
  scale_fill_manual(values = Palette4)+
  scale_color_manual(values = Palette4)+
  guides(fill=guide_legend(title="Model variable set"),
         color=guide_legend(title="Model variable set")) +
  theme_minimal(base_family = 'Roboto Condensed') +
  facet_grid(cols = vars(Grouping), scales = 'free_x', switch = 'x', space = 'free_x') +
  theme(
    axis.text.x = element_text(size = 6, color = "black",margin = margin(0,0,0,0)),
    axis.text.y = element_text(size = 6, color = "black",margin = margin(0,0,0,0)),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold',margin = margin(0,0,0,0)),
    panel.border = element_blank(),
    axis.line.x = element_line(size=1/.pt),
    axis.text = element_text(color='black'),
    legend.position = 'bottom',
    panel.spacing = unit(5, 'points'),
    legend.key.size = unit(1.3/.pt,'line'),
    legend.title = element_text(size = 7, color = "black", face = 'bold'),
    legend.text=element_text(size=6),
    plot.margin=grid::unit(c(0,2,0,0), "mm"),
    strip.text = element_blank(),
    legend.margin=margin(0,0,0,0)
  )

# Create directory for current date and save TILBasic explanation over days of ICU stay at all points plot
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'all_point_somers.svg'),all.point.Somers.D.plot,device= svglite,units='in',dpi=600,width=3.7,height = 2.5)

## Plot and save TILBasic explanation over days of ICU stay at all points
# Create ggplot object
all.point.Somers.D.plot <- Somers.D.plot.df %>% ggplot() +
  geom_ribbon(data=Somers.D.plot.df %>% filter(Grouping==1),
              mapping=aes(x=ICUDay, ymin=100*lo, ymax=100*hi, fill=VariableSet, group = VariableSet),
              alpha=.2) +
  geom_line(data=Somers.D.plot.df %>% filter(Grouping==1),
            mapping=aes(x=ICUDay, y=100*median, color=VariableSet, group = VariableSet),
            lwd=1.75/.pt) +
  geom_errorbar(data=Somers.D.plot.df %>% filter(Grouping!=1),
                mapping=aes(x=ICUDay, ymin=100*lo, ymax=100*hi, color=VariableSet),
                position = position_dodge(width = .75),
                width=.35) +
  geom_point(data=Somers.D.plot.df %>% filter(Grouping!=1),
             mapping=aes(x=ICUDay, y=100*median, color=VariableSet),
             position = position_dodge(width = .75),
             size=1) +
  coord_cartesian(ylim = c(0,100)) +
  xlab("Day of ICU stay")+
  ylab("Explanation of ordinal variance in next-day TILBasic (%)")+
  scale_x_discrete(expand = expansion(mult = c(.05, .05)))+
  scale_y_continuous(breaks = seq(0,100,10)) +
  scale_fill_manual(values = Palette4)+
  scale_color_manual(values = Palette4)+
  guides(fill=guide_legend(title="Model variable set"),
         color=guide_legend(title="Model variable set")) +
  theme_minimal(base_family = 'Roboto Condensed') +
  facet_grid(cols = vars(Grouping), scales = 'free_x', switch = 'x', space = 'free_x') +
  theme(
    axis.text.x = element_text(size = 6, color = "black",margin = margin(0,0,0,0)),
    axis.text.y = element_text(size = 6, color = "black",margin = margin(0,0,0,0)),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold',margin = margin(0,0,0,0)),
    panel.border = element_blank(),
    axis.line.x = element_line(size=1/.pt),
    axis.text = element_text(color='black'),
    legend.position = 'bottom',
    panel.spacing = unit(5, 'points'),
    legend.key.size = unit(1.3/.pt,'line'),
    legend.title = element_text(size = 7, color = "black", face = 'bold'),
    legend.text=element_text(size=6),
    plot.margin=grid::unit(c(0,2,0,0), "mm"),
    strip.text = element_blank(),
    legend.margin=margin(0,0,0,0)
  )

# Create directory for current date and save TILBasic explanation over days of ICU stay at all points plot
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'all_point_somers.svg'),all.point.Somers.D.plot,device= svglite,units='in',dpi=600,width=3.7,height = 2.5)

## Plot and save TILBasic explanation over days of ICU stay at points of transition
# Create ggplot object
trans.point.Somers.D.plot <- trans.Somers.D.plot.df %>% ggplot() +
  geom_ribbon(data=trans.Somers.D.plot.df %>% filter(Grouping==1),
              mapping=aes(x=ICUDay, ymin=100*lo, ymax=100*hi, fill=VariableSet, group = VariableSet),
              alpha=.2) +
  geom_line(data=trans.Somers.D.plot.df %>% filter(Grouping==1),
            mapping=aes(x=ICUDay, y=100*median, color=VariableSet, group = VariableSet),
            lwd=1.75/.pt) +
  geom_errorbar(data=trans.Somers.D.plot.df %>% filter(Grouping!=1),
                mapping=aes(x=ICUDay, ymin=100*lo, ymax=100*hi, color=VariableSet),
                position = position_dodge(width = .75),
                width=.35) +
  geom_point(data=trans.Somers.D.plot.df %>% filter(Grouping!=1),
             mapping=aes(x=ICUDay, y=100*median, color=VariableSet),
             position = position_dodge(width = .75),
             size=1) +
  coord_cartesian(ylim = c(0,100)) +
  xlab("Day of ICU stay")+
  ylab("Explanation of ordinal variance in next-day TILBasic (%)")+
  scale_x_discrete(expand = expansion(mult = c(.05, .05)))+
  scale_y_continuous(breaks = seq(0,100,10)) +
  scale_fill_manual(values = Palette4)+
  scale_color_manual(values = Palette4)+
  guides(fill=guide_legend(title="Model variable set"),
         color=guide_legend(title="Model variable set")) +
  theme_minimal(base_family = 'Roboto Condensed') +
  facet_grid(cols = vars(Grouping), scales = 'free_x', switch = 'x', space = 'free_x') +
  theme(
    axis.text.x = element_text(size = 6, color = "black",margin = margin(0,0,0,0)),
    axis.text.y = element_text(size = 6, color = "black",margin = margin(0,0,0,0)),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold',margin = margin(0,0,0,0)),
    panel.border = element_blank(),
    axis.line.x = element_line(size=1/.pt),
    axis.text = element_text(color='black'),
    legend.position = 'bottom',
    panel.spacing = unit(5, 'points'),
    legend.key.size = unit(1.3/.pt,'line'),
    legend.title = element_text(size = 7, color = "black", face = 'bold'),
    legend.text=element_text(size=6),
    plot.margin=grid::unit(c(0,2,0,0), "mm"),
    strip.text = element_blank(),
    legend.margin=margin(0,0,0,0)
  )

# Create directory for current date and save TILBasic explanation over days of ICU stay at points of transition
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'trans_point_somers.svg'),trans.point.Somers.D.plot,device= svglite,units='in',dpi=600,width=3.7,height = 2.5)















### Ordinal c-index in next-day TILBasic prediction
## Load and prepare dataframes for explanation at all points
# Load full-model ORC at all points
TIL.Basic.ORC.CIs <- read.csv('../TILTomorrow_model_performance/v2-0/test_set_metrics_CI.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter(METRIC=='ORC',
         TUNE_IDX==332) %>%
  mutate(ICUDay = sprintf('Day %.0f',WINDOW_IDX),
         Grouping = case_when(WINDOW_IDX<=6~'1',
                              WINDOW_IDX<=9~'2',
                              WINDOW_IDX<=13~'3'),
         ICUDay=fct_reorder(factor(ICUDay), WINDOW_IDX),
         VariableSet = 'Full')

# Load sensitivity-analysis ORC at all points
sens.analysis.ORC.CIs <- read.csv('../TILTomorrow_model_performance/v2-0/sens_analysis_metrics_CI.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter(METRIC=='ORC',
         TUNE_IDX==332,
         SENS_IDX %in% c(1,4)) %>%
  mutate(ICUDay = sprintf('Day %.0f',WINDOW_IDX),
         Grouping = case_when(WINDOW_IDX<=6~'1',
                              WINDOW_IDX<=9~'2',
                              WINDOW_IDX<=13~'3'),
         ICUDay=fct_reorder(factor(ICUDay), WINDOW_IDX),
         VariableSet = case_when(SENS_IDX==1~'No dynamic',
                                 SENS_IDX==4~'No clinician impressions or treatments'))

# Load no-information ORC at all points
no.info.ORC.CIs <- read.csv('../TILTomorrow_model_performance/v2-0/no_information_metrics_CI.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter(METRIC=='ORC',
         TUNE_IDX==332) %>%
  mutate(ICUDay = sprintf('Day %.0f',WINDOW_IDX),
         Grouping = case_when(WINDOW_IDX<=6~'1',
                              WINDOW_IDX<=9~'2',
                              WINDOW_IDX<=13~'3'),
         ICUDay=fct_reorder(factor(ICUDay), WINDOW_IDX),
         VariableSet = 'Only last available TILBasic')

# Combine all-point ORC values into single dataframe
ORC.plot.df <- TIL.Basic.ORC.CIs %>%
  select(TUNE_IDX,ICUDay,lo,median,hi,Grouping,VariableSet) %>%
  rbind(sens.analysis.ORC.CIs %>%
          select(TUNE_IDX,ICUDay,lo,median,hi,Grouping,VariableSet)) %>%
  rbind(no.info.ORC.CIs %>%
          select(TUNE_IDX,ICUDay,lo,median,hi,Grouping,VariableSet)) %>%
  mutate(VariableSet = factor(VariableSet,levels=c('Full','Only last available TILBasic','No clinician impressions or treatments','No dynamic'))) %>%
  mutate(across(lo:hi, ~ (abs(.x)+.x)/2))

## Load and prepare dataframes for ORC at point of transition
# Load full-model ORC at point of transition
trans.TIL.Basic.ORC.CIs <- read.csv('../TILTomorrow_model_performance/v2-0/trans_test_set_metrics_CI.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter(METRIC=='ORC',
         TUNE_IDX==332) %>%
  mutate(ICUDay = sprintf('Day %.0f',WINDOW_IDX),
         Grouping = case_when(WINDOW_IDX<=6~'1',
                              WINDOW_IDX<=9~'2',
                              WINDOW_IDX<=13~'3'),
         ICUDay=fct_reorder(factor(ICUDay), WINDOW_IDX),
         VariableSet = 'Full')

# Load sensitivity-analysis ORC at point of transition
trans.sens.analysis.ORC.CIs <- read.csv('../TILTomorrow_model_performance/v2-0/trans_sens_analysis_metrics_CI.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter(METRIC=='ORC',
         TUNE_IDX==332,
         SENS_IDX %in% c(1,4)) %>%
  mutate(ICUDay = sprintf('Day %.0f',WINDOW_IDX),
         Grouping = case_when(WINDOW_IDX<=6~'1',
                              WINDOW_IDX<=9~'2',
                              WINDOW_IDX<=13~'3'),
         ICUDay=fct_reorder(factor(ICUDay), WINDOW_IDX),
         VariableSet = case_when(SENS_IDX==1~'No dynamic',
                                 SENS_IDX==4~'No clinician impressions or treatments'))

# Load no-information ORC at point of transition
trans.no.info.ORC.CIs <- read.csv('../TILTomorrow_model_performance/v2-0/trans_no_information_metrics_CI.csv',na.strings = c("NA","NaN","", " ")) %>%
  filter(METRIC=='ORC',
         TUNE_IDX==332) %>%
  mutate(ICUDay = sprintf('Day %.0f',WINDOW_IDX),
         Grouping = case_when(WINDOW_IDX<=6~'1',
                              WINDOW_IDX<=9~'2',
                              WINDOW_IDX<=13~'3'),
         ICUDay=fct_reorder(factor(ICUDay), WINDOW_IDX),
         VariableSet = 'Only last available TILBasic')

# Combine transition-point ORC values into single dataframe
trans.ORC.plot.df <- trans.TIL.Basic.ORC.CIs %>%
  select(TUNE_IDX,ICUDay,lo,median,hi,Grouping,VariableSet) %>%
  rbind(trans.sens.analysis.ORC.CIs %>%
          select(TUNE_IDX,ICUDay,lo,median,hi,Grouping,VariableSet)) %>%
  rbind(trans.no.info.ORC.CIs %>%
          select(TUNE_IDX,ICUDay,lo,median,hi,Grouping,VariableSet)) %>%
  mutate(VariableSet = factor(VariableSet,levels=c('Full','Only last available TILBasic','No clinician impressions or treatments','No dynamic'))) %>%
  mutate(across(lo:hi, ~ (abs(.x)+.x)/2))

## Plot and save TILBasic ORC over days of ICU stay at all points
# Create ggplot object
all.point.ORC.plot <- ORC.plot.df %>% ggplot() +
  geom_line(data=ORC.plot.df %>% filter(Grouping==1),
            mapping=aes(x=ICUDay, y=100*median, color=VariableSet, group = VariableSet),
            lwd=1.3/.pt) +
  geom_ribbon(data=ORC.plot.df %>% filter(Grouping==1),
              mapping=aes(x=ICUDay, ymin=100*lo, ymax=100*hi, fill=VariableSet, group = VariableSet),
              alpha=.2) +
  geom_point(data=ORC.plot.df %>% filter(Grouping!=1),
             mapping=aes(x=ICUDay, y=100*median, color=VariableSet),
             position = position_dodge(width = .75),
             size=1) +
  geom_errorbar(data=ORC.plot.df %>% filter(Grouping!=1),
                mapping=aes(x=ICUDay, ymin=100*lo, ymax=100*hi, color=VariableSet),
                position = position_dodge(width = .75),
                width=.35) +
  coord_cartesian(ylim = c(0,100)) +
  xlab("Day of ICU stay")+
  ylab("ORC")+
  scale_y_continuous(breaks = seq(0,100,10)) +
  scale_fill_manual(values = Palette4)+
  scale_color_manual(values = Palette4)+
  guides(fill=guide_legend(title="Model variable set"),
         color=guide_legend(title="Model variable set")) +
  theme_minimal(base_family = 'Roboto Condensed') +
  facet_grid(cols = vars(Grouping), scales = 'free_x', switch = 'x', space = 'free_x') +
  theme(
    axis.text.x = element_text(size = 6, color = "black",margin = margin(0,0,0,0)),
    axis.text.y = element_text(size = 6, color = "black",margin = margin(0,0,0,0)),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold',margin = margin(0,0,0,0)),
    panel.border = element_blank(),
    axis.line.x = element_line(size=1/.pt),
    axis.text = element_text(color='black'),
    legend.position = 'bottom',
    panel.spacing = unit(5, 'points'),
    legend.key.size = unit(1.3/.pt,'line'),
    legend.title = element_text(size = 7, color = "black", face = 'bold'),
    legend.text=element_text(size=6),
    plot.margin=grid::unit(c(0,2,0,0), "mm"),
    strip.text = element_blank(),
    legend.margin=margin(0,0,0,0)
  )

# Create directory for current date and save TILBasic ORC over days of ICU stay at all points plot
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'all_point_ORC.svg'),all.point.ORC.plot,device= svglite,units='in',dpi=600,width=3.7,height = 2.5)

## Plot and save TILBasic ORC over days of ICU stay at all points
# Create ggplot object
all.point.ORC.plot <- ORC.plot.df %>% ggplot() +
  geom_ribbon(data=ORC.plot.df %>% filter(Grouping==1),
              mapping=aes(x=ICUDay, ymin=100*lo, ymax=100*hi, fill=VariableSet, group = VariableSet),
              alpha=.2) +
  geom_line(data=ORC.plot.df %>% filter(Grouping==1),
            mapping=aes(x=ICUDay, y=100*median, color=VariableSet, group = VariableSet),
            lwd=1.75/.pt) +
  geom_errorbar(data=ORC.plot.df %>% filter(Grouping!=1),
                mapping=aes(x=ICUDay, ymin=100*lo, ymax=100*hi, color=VariableSet),
                position = position_dodge(width = .75),
                width=.35) +
  geom_point(data=ORC.plot.df %>% filter(Grouping!=1),
             mapping=aes(x=ICUDay, y=100*median, color=VariableSet),
             position = position_dodge(width = .75),
             size=1) +
  coord_cartesian(ylim = c(0,100)) +
  xlab("Day of ICU stay")+
  ylab("ORC of ordinal variance in next-day TILBasic (%)")+
  scale_x_discrete(expand = expansion(mult = c(.05, .05)))+
  scale_y_continuous(breaks = seq(0,100,10)) +
  scale_fill_manual(values = Palette4)+
  scale_color_manual(values = Palette4)+
  guides(fill=guide_legend(title="Model variable set"),
         color=guide_legend(title="Model variable set")) +
  theme_minimal(base_family = 'Roboto Condensed') +
  facet_grid(cols = vars(Grouping), scales = 'free_x', switch = 'x', space = 'free_x') +
  theme(
    axis.text.x = element_text(size = 6, color = "black",margin = margin(0,0,0,0)),
    axis.text.y = element_text(size = 6, color = "black",margin = margin(0,0,0,0)),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold',margin = margin(0,0,0,0)),
    panel.border = element_blank(),
    axis.line.x = element_line(size=1/.pt),
    axis.text = element_text(color='black'),
    legend.position = 'bottom',
    panel.spacing = unit(5, 'points'),
    legend.key.size = unit(1.3/.pt,'line'),
    legend.title = element_text(size = 7, color = "black", face = 'bold'),
    legend.text=element_text(size=6),
    plot.margin=grid::unit(c(0,2,0,0), "mm"),
    strip.text = element_blank(),
    legend.margin=margin(0,0,0,0)
  )

# Create directory for current date and save TILBasic ORC over days of ICU stay at all points plot
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'all_point_ORC.svg'),all.point.ORC.plot,device= svglite,units='in',dpi=600,width=3.7,height = 2.5)

## Plot and save TILBasic ORC over days of ICU stay at points of transition
# Create ggplot object
trans.point.ORC.plot <- trans.ORC.plot.df %>% ggplot() +
  geom_ribbon(data=trans.ORC.plot.df %>% filter(Grouping==1),
              mapping=aes(x=ICUDay, ymin=100*lo, ymax=100*hi, fill=VariableSet, group = VariableSet),
              alpha=.2) +
  geom_line(data=trans.ORC.plot.df %>% filter(Grouping==1),
            mapping=aes(x=ICUDay, y=100*median, color=VariableSet, group = VariableSet),
            lwd=1.75/.pt) +
  geom_errorbar(data=trans.ORC.plot.df %>% filter(Grouping!=1),
                mapping=aes(x=ICUDay, ymin=100*lo, ymax=100*hi, color=VariableSet),
                position = position_dodge(width = .75),
                width=.35) +
  geom_point(data=trans.ORC.plot.df %>% filter(Grouping!=1),
             mapping=aes(x=ICUDay, y=100*median, color=VariableSet),
             position = position_dodge(width = .75),
             size=1) +
  coord_cartesian(ylim = c(0,100)) +
  xlab("Day of ICU stay")+
  ylab("ORC of ordinal variance in next-day TILBasic (%)")+
  scale_x_discrete(expand = expansion(mult = c(.05, .05)))+
  scale_y_continuous(breaks = seq(0,100,10)) +
  scale_fill_manual(values = Palette4)+
  scale_color_manual(values = Palette4)+
  guides(fill=guide_legend(title="Model variable set"),
         color=guide_legend(title="Model variable set")) +
  theme_minimal(base_family = 'Roboto Condensed') +
  facet_grid(cols = vars(Grouping), scales = 'free_x', switch = 'x', space = 'free_x') +
  theme(
    axis.text.x = element_text(size = 6, color = "black",margin = margin(0,0,0,0)),
    axis.text.y = element_text(size = 6, color = "black",margin = margin(0,0,0,0)),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold',margin = margin(0,0,0,0)),
    panel.border = element_blank(),
    axis.line.x = element_line(size=1/.pt),
    axis.text = element_text(color='black'),
    legend.position = 'bottom',
    panel.spacing = unit(5, 'points'),
    legend.key.size = unit(1.3/.pt,'line'),
    legend.title = element_text(size = 7, color = "black", face = 'bold'),
    legend.text=element_text(size=6),
    plot.margin=grid::unit(c(0,2,0,0), "mm"),
    strip.text = element_blank(),
    legend.margin=margin(0,0,0,0)
  )

# Create directory for current date and save TILBasic ORC over days of ICU stay at points of transition
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'trans_point_ORC.svg'),trans.point.ORC.plot,device= svglite,units='in',dpi=600,width=3.7,height = 2.5)

## TomorrowTILBasic results


TIL.Basic.calib.curves.CIs <- read.csv('../TILTomorrow_model_performance/v2-0/test_set_calibration_curves_CI.csv',na.strings = c("NA","NaN","", " ")) %>%
  mutate(Tomorrow = sprintf('Day %.0f',WINDOW_IDX+1),
         Grouping = case_when(WINDOW_IDX<=6~'1',
                              WINDOW_IDX<=9~'2',
                              WINDOW_IDX<=13~'3',
                              WINDOW_IDX<=20~'4'),
         Tomorrow=fct_reorder(factor(Tomorrow), WINDOW_IDX),
         VariableSet = 'full')

##


sens.analysis.diff.metrics.CIs <- read.csv('../TILTomorrow_model_performance/v2-0/sens_analysis_metrics_diff_CI.csv',na.strings = c("NA","NaN","", " ")) %>%
  mutate(Tomorrow = sprintf('Day %.0f',WINDOW_IDX+1),
         Grouping = case_when(WINDOW_IDX<=6~'1',
                              WINDOW_IDX<=9~'2',
                              WINDOW_IDX<=13~'3',
                              WINDOW_IDX<=20~'4'),
         Tomorrow=fct_reorder(factor(Tomorrow), WINDOW_IDX),
         VariableSet = case_when(SENS_IDX==1~'w/o dynamic',
                                 SENS_IDX==2~'w/o clinician impressions',
                                 SENS_IDX==3~'w/o treatments',
                                 SENS_IDX==4~'w/o clinician impressions and treatments'))

sens.analysis.calib.curves.CIs <- read.csv('../TILTomorrow_model_performance/v2-0/sens_analysis_calibration_curves_CI.csv',na.strings = c("NA","NaN","", " ")) %>%
  mutate(Tomorrow = sprintf('Day %.0f',WINDOW_IDX+1),
         Grouping = case_when(WINDOW_IDX<=6~'1',
                              WINDOW_IDX<=9~'2',
                              WINDOW_IDX<=13~'3',
                              WINDOW_IDX<=20~'4'),
         Tomorrow=fct_reorder(factor(Tomorrow), WINDOW_IDX),
         VariableSet = case_when(SENS_IDX==1~'w/o dynamic',
                                 SENS_IDX==2~'w/o clinician impressions',
                                 SENS_IDX==3~'w/o treatments',
                                 SENS_IDX==4~'w/o clinician impressions and treatments'))

## ORC plot
ORC.plot.df <- TIL.Basic.CIs %>%
  filter(METRIC=='ORC',
         TUNE_IDX==332) %>%
  select(TUNE_IDX,Tomorrow,lo,median,hi,Grouping,VariableSet) %>%
  rbind(sens.analysis.metrics.CIs %>%
          filter(METRIC=='ORC',
                 TUNE_IDX==332) %>%
          select(TUNE_IDX,Tomorrow,lo,median,hi,Grouping,VariableSet)) %>%
  mutate(VariableSet = factor(VariableSet,levels=c('full','w/o dynamic','w/o clinician impressions','w/o treatments','w/o clinician impressions and treatments')))

TIL.Basic.ORC.plot <- ggplot() +
  geom_line(data=ORC.plot.df %>% filter(Grouping==1),
            mapping=aes(x=Tomorrow, y=median, color=VariableSet, group = VariableSet),
            lwd=1.3) +
  geom_ribbon(data=ORC.plot.df %>% filter(Grouping==1),
              mapping=aes(x=Tomorrow, ymin=lo, ymax=hi, fill=VariableSet, group = VariableSet),
              alpha=.2) +
  geom_point(data=ORC.plot.df %>% filter(Grouping!=1),
             mapping=aes(x=Tomorrow, y=median, color=VariableSet),
             position = position_dodge(width = .75),
             size=2) +
  geom_errorbar(data=ORC.plot.df %>% filter(Grouping!=1),
                mapping=aes(x=Tomorrow, ymin=lo, ymax=hi, color=VariableSet),
                position = position_dodge(width = .75),
                width=.35) +
  coord_cartesian(ylim = c(0.5,1)) +
  xlab("Tomorrow")+
  ylab("Ordinal c-index")+
  scale_y_continuous(breaks = seq(0.5,1,.1)) +
  scale_x_discrete(limits=levels(TIL.Basic.CIs$Tomorrow)) +
  theme_minimal(base_family = 'Roboto Condensed') +
  theme(
    axis.title.y = element_text(color = "black",face = 'bold'),
    axis.text.x = element_text(color = 'black'),
    axis.text.y = element_text(color = 'black'),
    axis.title.x = element_text(color = "black",face = 'bold'),
    panel.border = element_blank(),
    axis.line.x = element_line(size=1/.pt),
    axis.text = element_text(color='black'),
    legend.position = 'bottom',
    panel.spacing = unit(10, 'points'),
    legend.key.size = unit(1.3/.pt,'line'),
    legend.title = element_text(color = 'black',face = 'bold'),
    # legend.text=element_text(size=6),
    plot.margin=grid::unit(c(0,2,0,0), "mm"),
    strip.text = element_text(color = "black",face = 'bold'),
    legend.margin=margin(0,0,0,0)
  )

## Somers.D plot
Somers.D.plot.df <- TIL.Basic.CIs %>%
  filter(METRIC=='Somers D',
         TUNE_IDX==332) %>%
  select(TUNE_IDX,Tomorrow,lo,median,hi,Grouping,VariableSet) %>%
  rbind(sens.analysis.metrics.CIs %>%
          filter(METRIC=='Somers D',
                 TUNE_IDX==332) %>%
          select(TUNE_IDX,Tomorrow,lo,median,hi,Grouping,VariableSet)) %>%
  mutate(VariableSet = factor(VariableSet,levels=c('full','w/o dynamic','w/o clinician impressions','w/o treatments','w/o clinician impressions and treatments')))

TIL.Basic.Somers.D.plot <- ggplot() +
  geom_line(data=Somers.D.plot.df %>% filter(Grouping==1),
            mapping=aes(x=Tomorrow, y=100*median, color=VariableSet, group = VariableSet),
            lwd=1.3) +
  geom_ribbon(data=Somers.D.plot.df %>% filter(Grouping==1),
              mapping=aes(x=Tomorrow, ymin=100*lo, ymax=100*hi, fill=VariableSet, group = VariableSet),
              alpha=.2) +
  geom_point(data=Somers.D.plot.df %>% filter(Grouping!=1),
             mapping=aes(x=Tomorrow, y=100*median, color=VariableSet),
             position = position_dodge(width = .75),
             size=2) +
  geom_errorbar(data=Somers.D.plot.df %>% filter(Grouping!=1),
                mapping=aes(x=Tomorrow, ymin=100*lo, ymax=100*hi, color=VariableSet),
                position = position_dodge(width = .75),
                width=.35) +
  coord_cartesian(ylim = c(0,100)) +
  xlab("Tomorrow")+
  ylab("Explanation of ordinal variance in TILBasic (%)")+
  scale_y_continuous(breaks = seq(0,100,10)) +
  scale_x_discrete(limits=levels(Somers.D.plot.df$Tomorrow)) +
  scale_fill_manual(values = c('#003f5c','#58508d','#bc5090','#ff6361','#ffa600'))+
  scale_color_manual(values = c('#003f5c','#58508d','#bc5090','#ff6361','#ffa600'))+
  theme_minimal(base_family = 'Roboto Condensed') +
  theme(
    axis.title.y = element_text(color = "black",face = 'bold'),
    axis.text.x = element_text(color = 'black'),
    axis.text.y = element_text(color = 'black'),
    axis.title.x = element_text(color = "black",face = 'bold'),
    panel.border = element_blank(),
    axis.line.x = element_line(size=1/.pt),
    axis.text = element_text(color='black'),
    legend.position = 'bottom',
    panel.spacing = unit(10, 'points'),
    legend.key.size = unit(1.3/.pt,'line'),
    legend.title = element_text(color = 'black',face = 'bold'),
    # legend.text=element_text(size=6),
    plot.margin=grid::unit(c(0,2,0,0), "mm"),
    strip.text = element_text(color = "black",face = 'bold'),
    legend.margin=margin(0,0,0,0)
  )

## Difference in Somers.D plot
diff.Somers.D.plot.df <- sens.analysis.diff.metrics.CIs %>%
  filter(METRIC=='Somers D',
         TUNE_IDX==332) %>%
  select(TUNE_IDX,Tomorrow,lo,median,hi,Grouping,VariableSet) %>%
  mutate(VariableSet = factor(VariableSet,levels=c('w/o dynamic','w/o clinician impressions','w/o treatments','w/o clinician impressions and treatments')))

TIL.Basic.diff.Somers.D.plot <- ggplot() +
  geom_line(data=diff.Somers.D.plot.df %>% filter(Grouping==1),
            mapping=aes(x=Tomorrow, y=100*median, color=VariableSet, group = VariableSet),
            lwd=1.3) +
  geom_ribbon(data=diff.Somers.D.plot.df %>% filter(Grouping==1),
              mapping=aes(x=Tomorrow, ymin=100*lo, ymax=100*hi, fill=VariableSet, group = VariableSet),
              alpha=.2) +
  geom_point(data=diff.Somers.D.plot.df %>% filter(Grouping!=1),
             mapping=aes(x=Tomorrow, y=100*median, color=VariableSet),
             position = position_dodge(width = .75),
             size=2) +
  geom_errorbar(data=diff.Somers.D.plot.df %>% filter(Grouping!=1),
                mapping=aes(x=Tomorrow, ymin=100*lo, ymax=100*hi, color=VariableSet),
                position = position_dodge(width = .75),
                width=.35) +
  coord_cartesian(ylim = c(0,100)) +
  scale_fill_manual(values = c('#58508d','#bc5090','#ff6361','#ffa600'))+
  scale_color_manual(values = c('#58508d','#bc5090','#ff6361','#ffa600'))+
  xlab("Tomorrow")+
  ylab('Added explanation of ordinal variance in TILBasic (d%)')+
  scale_y_continuous(breaks = seq(0,100,10)) +
  scale_x_discrete(limits=levels(diff.Somers.D.plot.df$Tomorrow)) +
  theme_minimal(base_family = 'Roboto Condensed') +
  theme(
    axis.title.y = element_text(color = "black",face = 'bold'),
    axis.text.x = element_text(color = 'black'),
    axis.text.y = element_text(color = 'black'),
    axis.title.x = element_text(color = "black",face = 'bold'),
    panel.border = element_blank(),
    axis.line.x = element_line(size=1/.pt),
    axis.text = element_text(color='black'),
    legend.position = 'bottom',
    panel.spacing = unit(10, 'points'),
    legend.key.size = unit(1.3/.pt,'line'),
    legend.title = element_text(color = 'black',face = 'bold'),
    # legend.text=element_text(size=6),
    plot.margin=grid::unit(c(0,2,0,0), "mm"),
    strip.text = element_text(color = "black",face = 'bold'),
    legend.margin=margin(0,0,0,0)
  )

## TILBasic AUCs
AUC.plot.df <- TIL.Basic.CIs %>%
  filter(METRIC=='AUC',
         TUNE_IDX==332) %>%
  select(TUNE_IDX,Tomorrow,THRESHOLD,lo,median,hi,Grouping,VariableSet) %>%
  rbind(sens.analysis.metrics.CIs %>%
          filter(METRIC=='AUC',
                 TUNE_IDX==332) %>%
          select(TUNE_IDX,Tomorrow,THRESHOLD,lo,median,hi,Grouping,VariableSet)) %>%
  mutate(VariableSet = factor(VariableSet,levels=c('full','w/o dynamic','w/o clinician impressions','w/o treatments','w/o clinician impressions and treatments')))

TIL.Basic.AUC.plot <- ggplot() +
  geom_line(data=AUC.plot.df %>% filter(Grouping==1),
            mapping=aes(x=Tomorrow, y=median, color=VariableSet, group = VariableSet),
            lwd=1.3) +
  geom_ribbon(data=AUC.plot.df %>% filter(Grouping==1),
              mapping=aes(x=Tomorrow, ymin=lo, ymax=hi, fill=VariableSet, group = VariableSet),
              alpha=.2) +
  geom_point(data=AUC.plot.df %>% filter(Grouping!=1),
             mapping=aes(x=Tomorrow, y=median, color=VariableSet),
             position = position_dodge(width = .75),
             size=2) +
  geom_errorbar(data=AUC.plot.df %>% filter(Grouping!=1),
                mapping=aes(x=Tomorrow, ymin=lo, ymax=hi, color=VariableSet),
                position = position_dodge(width = .75),
                width=.35) +
  facet_wrap(~THRESHOLD,
             nrow=1,
             scales = 'free')+
  coord_cartesian(ylim = c(0.5,1)) +
  scale_fill_manual(values = c('#003f5c','#58508d','#bc5090','#ff6361','#ffa600'))+
  scale_color_manual(values = c('#003f5c','#58508d','#bc5090','#ff6361','#ffa600'))+
  xlab("Tomorrow")+
  ylab("AUC")+
  scale_y_continuous(breaks = seq(0.5,1,.1)) +
  scale_x_discrete(limits=levels(AUC.plot.df$Tomorrow)) +
  theme_minimal(base_family = 'Roboto Condensed') +
  theme(
    axis.title.y = element_text(color = "black",face = 'bold'),
    axis.text.x = element_text(color = 'black'),
    axis.text.y = element_text(color = 'black'),
    axis.title.x = element_text(color = "black",face = 'bold'),
    panel.border = element_blank(),
    axis.line.x = element_line(size=1/.pt),
    axis.text = element_text(color='black'),
    legend.position = 'bottom',
    # legend.position = 'none',
    panel.spacing = unit(10, 'points'),
    legend.key.size = unit(1.3/.pt,'line'),
    legend.title = element_text(color = 'black',face = 'bold'),
    # legend.text=element_text(size=6),
    plot.margin=grid::unit(c(0,2,0,0), "mm"),
    strip.text = element_text(color = "black",face = 'bold'),
    legend.margin=margin(0,0,0,0)
  )

## TILBasic Difference in AUCs plot
diff.AUC.plot.df <- sens.analysis.diff.metrics.CIs %>%
  filter(METRIC=='AUC',
         TUNE_IDX==332) %>%
  select(TUNE_IDX,Tomorrow,THRESHOLD,lo,median,hi,Grouping,VariableSet) %>%
  mutate(VariableSet = factor(VariableSet,levels=c('w/o dynamic','w/o clinician impressions','w/o treatments','w/o clinician impressions and treatments')))

TIL.Basic.diff.AUC.plot <- ggplot() +
  geom_line(data=diff.AUC.plot.df %>% filter(Grouping==1),
            mapping=aes(x=Tomorrow, y=median, color=VariableSet, group = VariableSet),
            lwd=1.3) +
  geom_ribbon(data=diff.AUC.plot.df %>% filter(Grouping==1),
              mapping=aes(x=Tomorrow, ymin=lo, ymax=hi, fill=VariableSet, group = VariableSet),
              alpha=.2) +
  geom_point(data=diff.AUC.plot.df %>% filter(Grouping!=1),
             mapping=aes(x=Tomorrow, y=median, color=VariableSet),
             position = position_dodge(width = .75),
             size=2) +
  geom_errorbar(data=diff.AUC.plot.df %>% filter(Grouping!=1),
                mapping=aes(x=Tomorrow, ymin=lo, ymax=hi, color=VariableSet),
                position = position_dodge(width = .75),
                width=.35) +
  facet_wrap(~THRESHOLD,
             nrow=1,
             scales = 'free')+
  coord_cartesian(ylim = c(0,0.5)) +
  scale_fill_manual(values = c('#58508d','#bc5090','#ff6361','#ffa600'))+
  scale_color_manual(values = c('#58508d','#bc5090','#ff6361','#ffa600'))+
  xlab("Tomorrow")+
  ylab("AUC")+
  scale_y_continuous(breaks = seq(0,0.5,.1)) +
  scale_x_discrete(limits=levels(diff.AUC.plot.df$Tomorrow)) +
  theme_minimal(base_family = 'Roboto Condensed') +
  theme(
    axis.title.y = element_text(color = "black",face = 'bold'),
    axis.text.x = element_text(color = 'black'),
    axis.text.y = element_text(color = 'black'),
    axis.title.x = element_text(color = "black",face = 'bold'),
    panel.border = element_blank(),
    axis.line.x = element_line(size=1/.pt),
    axis.text = element_text(color='black'),
    legend.position = 'bottom',
    # legend.position = 'none',
    panel.spacing = unit(10, 'points'),
    legend.key.size = unit(1.3/.pt,'line'),
    legend.title = element_text(color = 'black',face = 'bold'),
    # legend.text=element_text(size=6),
    plot.margin=grid::unit(c(0,2,0,0), "mm"),
    strip.text = element_text(color = "black",face = 'bold'),
    legend.margin=margin(0,0,0,0)
  )

## TILBasic calibration curves plot
TIL.Basic.calib.curves <- TIL.Basic.calib.curves.CIs %>%
  filter(TUNE_IDX == 332,
         Tomorrow %in% c('Day 2','Day 3','Day 4','Day 5','Day 6','Day 7')) %>%
  ggplot(aes(x=100*PREDPROB)) +
  facet_wrap( ~ THRESHOLD,
              scales = 'free',
              ncol = 4) +
  coord_cartesian(ylim = c(0,100),xlim = c(0,100))+
  geom_segment(x = 0, y = 0, xend = 100, yend = 100,alpha = 0.5,linetype = "dashed",size=.75/.pt, color = 'gray')+
  geom_ribbon(aes(ymin = 100*lo, ymax = 100*hi, fill = Tomorrow), alpha = 0.3,size=.75/.pt,color=NA) +
  geom_line(aes(y = 100*median, color = Tomorrow), alpha = 1, size=1.3/.pt) +
  scale_x_continuous(expand = expansion(mult = c(.01, .01))) +
  scale_y_continuous(expand = expansion(mult = c(.01, .01))) +
  guides(fill=guide_legend(nrow=2,byrow=TRUE),color=guide_legend(nrow=2,byrow=TRUE)) +
  # scale_fill_manual(name = "Time since ICU admission",
  #                   values = c("#003f5c", "#7a5195", "#ef5675",'#ffa600'))+
  # scale_color_manual(name = "Time since ICU admission",
  #                    values = c("#003f5c", "#7a5195", "#ef5675",'#ffa600'))+
  xlab("Predicted probability") +
  ylab("Observed probability") +
  theme_classic(base_family = 'Roboto Condensed') +
  theme(
    strip.text = element_text(size=6, color = "black",face = 'bold',margin = margin(b = .5)), 
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    panel.spacing = unit(0.05, "lines"),
    axis.text.x = element_text(size = 5, color = "black",margin = margin(r = 0)),
    axis.text.y = element_text(size = 5, color = "black",margin = margin(r = 0)),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold'),
    strip.background = element_blank(),
    aspect.ratio = 1,
    panel.border = element_rect(colour = 'black', fill=NA, size = 1/.pt),
    #plot.margin=grid::unit(c(0,0,0,0), "mm"),
    legend.position = 'bottom',
    legend.title = element_text(size = 7, color = "black", face = 'bold'),
    legend.text=element_text(size=6),
    axis.line = element_blank(),
    legend.key.size = unit(1.3/.pt,"line")
  )

###########
### Relevance layer plots
## Baseline tokens
baseline.relevances.plot.df <- read.csv('../TILTomorrow_model_interpretations/v2-0/relevance_layer/baseline_relevances_plot_df.csv',na.strings = c('')) %>%
  arrange(TUNE_IDX,DROPOUT_VARS, -median) %>%
  group_by(TUNE_IDX,DROPOUT_VARS) %>% 
  mutate(PRED_RANK = rank(-median),
         Type = replace_na(Type,'Other'))
baseline.relevances.plot.df$GROUPS <- ''
baseline.relevances.plot.df$GROUPS[baseline.relevances.plot.df$PRED_RANK <= 20] <- 'Top'
baseline.relevances.plot.df$GROUPS[baseline.relevances.plot.df$PRED_RANK > 21] <- 'Bottom'
baseline.relevances.plot.df$GROUPS[baseline.relevances.plot.df$PRED_RANK == 21] <- 'Middle'

baseline.relevances.plot.df %>% 
  filter(TUNE_IDX==332,
         DROPOUT_VARS=='none') %>%
  relevance.boxplots

baseline.relevances.plot.df %>% 
  filter(TUNE_IDX==332,
         DROPOUT_VARS=='dynamic') %>%
  relevance.boxplots

## Dynamic tokens
dynamic.relevances.plot.df <- read.csv('../TILTomorrow_model_interpretations/v2-0/relevance_layer/dynamic_relevances_plot_df.csv',na.strings = c('')) %>%
  arrange(TUNE_IDX,DROPOUT_VARS, -median) %>%
  group_by(TUNE_IDX,DROPOUT_VARS) %>% 
  mutate(PRED_RANK = rank(-median)) %>%
  mutate(Type = replace_na(Type,'Other'))
dynamic.relevances.plot.df$GROUPS <- ''
dynamic.relevances.plot.df$GROUPS[dynamic.relevances.plot.df$PRED_RANK <= 20] <- 'Top'
dynamic.relevances.plot.df$GROUPS[dynamic.relevances.plot.df$PRED_RANK > 21] <- 'Bottom'
dynamic.relevances.plot.df$GROUPS[dynamic.relevances.plot.df$PRED_RANK == 21] <- 'Middle'

dynamic.relevances.plot.df %>% 
  filter(TUNE_IDX==332,
         DROPOUT_VARS=='none') %>%
  relevance.boxplots

dynamic.relevances.plot.df %>% 
  filter(TUNE_IDX==332,
         DROPOUT_VARS=='clinician_impressions_and_treatments') %>%
  relevance.boxplots

dynamic.relevances.plot.df %>% 
  filter(TUNE_IDX==332,
         DROPOUT_VARS=='treatments') %>%
  relevance.boxplots

## Intervention tokens
intervention.relevances.plot.df <- read.csv('../TILTomorrow_model_interpretations/v2-0/relevance_layer/intervention_relevances_plot_df.csv',na.strings = c('')) %>%
  arrange(TUNE_IDX, -median) %>%
  group_by(TUNE_IDX) %>% 
  mutate(PRED_RANK = rank(-median)) %>%
  mutate(Type = replace_na(Type,'Other'))
intervention.relevances.plot.df$GROUPS <- ''
intervention.relevances.plot.df$GROUPS[intervention.relevances.plot.df$PRED_RANK <= 20] <- 'Top'
intervention.relevances.plot.df$GROUPS[intervention.relevances.plot.df$PRED_RANK > 21] <- 'Bottom'
intervention.relevances.plot.df$GROUPS[intervention.relevances.plot.df$PRED_RANK == 21] <- 'Middle'

intervention.relevances.plot.df %>% 
  filter(TUNE_IDX==332,
         DROPOUT_VARS=='none') %>%
  relevance.boxplots

###########
### TimeSHAP plots
## Characterise filtered TimeSHAP dataframes constructed for plotting
filt.df.files <- list.files('../TILTomorrow_model_interpretations/v2-0/timeSHAP/viz_feature_values',full.names = T) %>%
  data.frame(FILE=.) %>%
  mutate(Dropout.Var = str_match(FILE, "dropout_\\s*(.*?)\\s*_timepoints_")[,2],
         Timepoints = str_match(FILE, "timepoints_\\s*(.*?)\\s*_subset_")[,2],
         Var.Subset = str_match(FILE, "_subset_\\s*(.*?)\\s*.csv")[,2]) %>%
  filter(Var.Subset!='nonmissing_type')

full.token.keys <- read_excel('../tokens/TILTomorrow_full_token_keys_v2-0.xlsx')

for (curr.idx in 1:nrow(filt.df.files)){
  
  curr.file <- filt.df.files$FILE[curr.idx]
  curr.drop.var <- filt.df.files$Dropout.Var[curr.idx]
  curr.timepoints <- filt.df.files$Timepoints[curr.idx]
  curr.var.subset <- filt.df.files$Var.Subset[curr.idx]
  
  filt.timeSHAP.df <- read.csv(curr.file,na.strings = c("NA","NaN","", " ")) %>%
    filter(BaselineFeatures=='Zero',
           TUNE_IDX==332) %>%
    mutate(Baseline = as.logical(Baseline),
           GROUPS = case_when((RankIdx >= 11) ~ 'Top',
                              (RankIdx <= 10) ~'Bottom'),
           GROUPS = factor(GROUPS,levels=c('Top','Middle','Bottom'))) %>%
    left_join(full.token.keys %>% select(Token,Ordered,Binary,OrderIdx)) %>%
    group_by(BaseToken) %>%
    mutate(TokenRankIdx=dense_rank(Token)) %>%
    ungroup() %>%
    mutate(TokenRankIdx = case_when((Ordered)|(Binary)~OrderIdx,
                                    T~(TokenRankIdx-1))) %>%
    arrange(BaselineFeatures,TUNE_IDX,Threshold,RankIdx,TokenRankIdx) %>%
    mutate(Baseline = recode(as.character(Baseline),'TRUE'='Static','FALSE'='Dynamic'),
           Baseline = fct_relevel(Baseline, 'Static', 'Dynamic'),
           PLOT_LABEL = fct_reorder(BaseToken, RankIdx))
  
  color.scale.calc.df <- filt.timeSHAP.df %>%
    select(BaseToken,Token,Binary,Ordered,TokenRankIdx) %>%
    unique() %>%
    group_by(BaseToken) %>%
    mutate(MaxOrderIdx = max(TokenRankIdx,na.rm = T)) %>%
    ungroup() %>%
    mutate(ColorScale = case_when(MaxOrderIdx>0~(TokenRankIdx)/(MaxOrderIdx),
                                  T~1),
           ColorScale = replace_na(ColorScale,-1)) 
  
  filt.timeSHAP.df <- filt.timeSHAP.df %>%
    left_join(color.scale.calc.df)
  
  # Create feature importance beeswarm plot for static predictors
  static.timeshap.plot <- filt.timeSHAP.df %>%
    filter(Baseline=='Static') %>%
    ggplot() +
    # coord_cartesian(xlim = c(-.1,.1)) +
    scale_color_gradient2(na.value='#488f31',low='#003f5c',mid='#eacaf4',high='#de425b',midpoint=.5,limits = c(0,1),breaks = c(0.05,.95), labels = c('Low','High')) +
    geom_vline(xintercept = 0, color = "darkgray") +
    geom_quasirandom(aes(y=PLOT_LABEL,x=SHAP,color=ColorScale),groupOnX=FALSE,varwidth = FALSE,alpha = 0.6,stroke = 0,size=1) + 
    theme_minimal(base_family = 'Roboto Condensed') +
    guides(color = guide_colourbar(title='Feature Value',title.vjust=1,barwidth = 10, barheight = .25,ticks = FALSE))+
    facet_grid(rows = vars(GROUPS), scales = 'free_y', switch = 'y', space = 'free_y') +
    theme(
      strip.background = element_blank(),
      strip.text = element_blank(),
      axis.title.y = element_blank(),
      axis.text.x = element_text(color = 'black'),
      axis.text.y = element_text(color = 'black',angle = 30, hjust=1),
      # axis.text.y = element_blank(),
      axis.title.x = element_blank(),
      panel.border = element_blank(),
      axis.line.x = element_line(size=1/.pt),
      axis.text = element_text(color='black'),
      legend.position = 'bottom',
      # legend.position = 'none',
      panel.grid.major.y = element_blank(),
      panel.spacing = unit(10, 'points'),
      legend.key.size = unit(1.3/.pt,'line'),
      legend.title = element_text(color = 'black',face = 'bold'),
      # legend.text=element_text(size=6),
      plot.margin=grid::unit(c(0,2,0,0), "mm")
    )
  
  # Create feature importance beeswarm plot for dynamic predictors
  dynamic.timeshap.plot <- filt.timeSHAP.df %>%
    filter(Baseline=='Dynamic') %>%
    ggplot() +
    # coord_cartesian(xlim = c(-.05,.05)) +
    scale_color_gradient2(na.value='#488f31',low='#003f5c',mid='#eacaf4',high='#de425b',midpoint=.5,limits = c(0,1),breaks = c(0.05,.95), labels = c('Low','High')) +
    geom_vline(xintercept = 0, color = "darkgray") +
    geom_quasirandom(aes(y=PLOT_LABEL,x=SHAP,color=ColorScale),groupOnX=FALSE,varwidth = FALSE,alpha = 0.6,stroke = 0,size=1) + 
    theme_minimal(base_family = 'Roboto Condensed') +
    guides(color = guide_colourbar(title='Feature Value',title.vjust=1,barwidth = 10, barheight = .25,ticks = FALSE))+
    facet_grid(rows = vars(GROUPS), scales = 'free_y', switch = 'y', space = 'free_y') +
    theme(
      strip.background = element_blank(),
      strip.text = element_blank(),
      axis.title.y = element_blank(),
      axis.text.x = element_text(color = 'black'),
      axis.text.y = element_text(color = 'black',angle = 30, hjust=1),
      # axis.text.y = element_blank(),
      axis.title.x = element_blank(),
      panel.border = element_blank(),
      axis.line.x = element_line(size=1/.pt),
      axis.text = element_text(color='black'),
      legend.position = 'bottom',
      # legend.position = 'none',
      panel.grid.major.y = element_blank(),
      panel.spacing = unit(10, 'points'),
      legend.key.size = unit(1.3/.pt,'line'),
      legend.title = element_text(color = 'black',face = 'bold'),
      # legend.text=element_text(size=6),
      plot.margin=grid::unit(c(0,2,0,0), "mm")
    )
  
  dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
  ggsave(file.path('../plots',Sys.Date(),paste0('dropout_',curr.drop.var,'_timepoints_',curr.timepoints,'_subset_',curr.var.subset,'_static.png')),static.timeshap.plot,units='px',width=4550,height=3925,dpi=600)
  ggsave(file.path('../plots',Sys.Date(),paste0('dropout_',curr.drop.var,'_timepoints_',curr.timepoints,'_subset_',curr.var.subset,'_dynamic.png')),dynamic.timeshap.plot,units='px',width=4550,height=3925,dpi=600)
}












######
## Prepare dataframe of filtered TimeSHAP values for plotting
# Load TimeSHAP value dataframe
filt.timeSHAP.df <- read.csv('../TILTomorrow_model_interpretations/v1-0/timeSHAP/filtered_plotting_timeSHAP_values.csv',
                             na.strings = c("NA","NaN","", " ")) %>%
  mutate(Baseline = as.logical(Baseline),
         Numeric = as.logical(Numeric),
         GROUPS = case_when((RankIdx >= 11) ~ 'Top',
                            (RankIdx <= 10) ~'Bottom'),
         GROUPS = factor(GROUPS,levels=c('Top','Middle','Bottom')),
         TokenRankIdx = case_when((Ordered=='True')|(Binary=='True')~OrderIdx,
                                  T~(TokenRankIdx-1))) %>%
  arrange(BaselineFeatures,TUNE_IDX,Threshold,RankIdx,TokenRankIdx) %>%
  mutate(Baseline = recode(as.character(Baseline),'TRUE'='Static','FALSE'='Dynamic'),
         Baseline = fct_relevel(Baseline, 'Static', 'Dynamic'),
         PLOT_LABEL = fct_reorder(BaseToken, RankIdx))

color.scale.calc.df <- filt.timeSHAP.df %>%
  select(BaseToken,Token,Binary,Ordered,TokenRankIdx) %>%
  unique() %>%
  group_by(BaseToken) %>%
  mutate(MaxOrderIdx = max(TokenRankIdx,na.rm = T)) %>%
  ungroup() %>%
  mutate(ColorScale = case_when(MaxOrderIdx>0~(TokenRankIdx)/(MaxOrderIdx),
                                T~1))

filt.timeSHAP.df <- filt.timeSHAP.df %>%
  left_join(color.scale.calc.df)

# Create feature importance beeswarm plot for static predictors
static.timeshap.plot <- filt.timeSHAP.df %>%
  filter(Baseline=='Static',
         TUNE_IDX==211,
         BaselineFeatures=='Zero') %>%
  mutate(ColorScale = replace_na(ColorScale,-1)) %>%
  ggplot() +
  coord_cartesian(xlim = c(-.1,.1)) +
  scale_color_gradient2(na.value='#488f31',low='#003f5c',mid='#eacaf4',high='#de425b',midpoint=.5,limits = c(0,1),breaks = c(0.05,.95), labels = c('Low','High')) +
  geom_vline(xintercept = 0, color = "darkgray") +
  geom_quasirandom(aes(y=PLOT_LABEL,x=SHAP,color=ColorScale),groupOnX=FALSE,varwidth = FALSE,alpha = 0.6,stroke = 0,size=1) + 
  theme_minimal(base_family = 'Roboto Condensed') +
  guides(color = guide_colourbar(title='Feature Value',title.vjust=1,barwidth = 10, barheight = .25,ticks = FALSE))+
  facet_grid(rows = vars(GROUPS), scales = 'free_y', switch = 'y', space = 'free_y') +
  theme(
    strip.background = element_blank(),
    strip.text = element_blank(),
    axis.title.y = element_blank(),
    axis.text.x = element_text(color = 'black'),
    axis.text.y = element_text(color = 'black',angle = 30, hjust=1),
    # axis.text.y = element_blank(),
    axis.title.x = element_blank(),
    panel.border = element_blank(),
    axis.line.x = element_line(size=1/.pt),
    axis.text = element_text(color='black'),
    legend.position = 'bottom',
    # legend.position = 'none',
    panel.grid.major.y = element_blank(),
    panel.spacing = unit(10, 'points'),
    legend.key.size = unit(1.3/.pt,'line'),
    legend.title = element_text(color = 'black',face = 'bold'),
    # legend.text=element_text(size=6),
    plot.margin=grid::unit(c(0,2,0,0), "mm")
  )

# Create feature importance beeswarm plot for dynamic predictors
dynamic.timeshap.plot <- filt.timeSHAP.df %>%
  filter(Baseline=='Dynamic',
         TUNE_IDX==211,
         BaselineFeatures=='Zero') %>%
  mutate(ColorScale = replace_na(ColorScale,-1)) %>%
  ggplot() +
  # coord_cartesian(xlim = c(-.05,.05)) +
  scale_color_gradient2(na.value='#488f31',low='#003f5c',mid='#eacaf4',high='#de425b',midpoint=.5,limits = c(0,1),breaks = c(0.05,.95), labels = c('Low','High')) +
  geom_vline(xintercept = 0, color = "darkgray") +
  geom_quasirandom(aes(y=PLOT_LABEL,x=SHAP,color=ColorScale),groupOnX=FALSE,varwidth = FALSE,alpha = 0.6,stroke = 0,size=1) + 
  theme_minimal(base_family = 'Roboto Condensed') +
  guides(color = guide_colourbar(title='Feature Value',title.vjust=1,barwidth = 10, barheight = .25,ticks = FALSE))+
  facet_grid(rows = vars(GROUPS), scales = 'free_y', switch = 'y', space = 'free_y') +
  theme(
    strip.background = element_blank(),
    strip.text = element_blank(),
    axis.title.y = element_blank(),
    axis.text.x = element_text(color = 'black'),
    axis.text.y = element_text(color = 'black',angle = 30, hjust=1),
    # axis.text.y = element_blank(),
    axis.title.x = element_blank(),
    panel.border = element_blank(),
    axis.line.x = element_line(size=1/.pt),
    axis.text = element_text(color='black'),
    legend.position = 'bottom',
    # legend.position = 'none',
    panel.grid.major.y = element_blank(),
    panel.spacing = unit(10, 'points'),
    legend.key.size = unit(1.3/.pt,'line'),
    legend.title = element_text(color = 'black',face = 'bold'),
    # legend.text=element_text(size=6),
    plot.margin=grid::unit(c(0,2,0,0), "mm")
  )




#######
types.timeSHAP.df <- read.csv('../TILTomorrow_model_interpretations/v1-0/timeSHAP/filtered_plotting_types_timeSHAP_values.csv',
                              na.strings = c("NA","NaN","", " ")) %>%
  filter(TUNE_IDX == 211,
         BaselineFeatures=='Zero') %>%
  mutate(Baseline = as.logical(Baseline),
         Numeric = as.logical(Numeric),
         Ordered = as.logical(Ordered),
         Binary = as.logical(Binary),
         GROUPS = case_when((RankIdx >= 11) ~ 'Top',
                            (RankIdx <= 10) ~'Bottom')) %>%
  mutate(GROUPS = factor(GROUPS,levels=c('Top','Middle','Bottom'))) %>%
  mutate(Baseline = recode(as.character(Baseline),'TRUE'='Static','FALSE'='Dynamic')) %>%
  mutate(Baseline = fct_relevel(Baseline, 'Static', 'Dynamic')) %>%
  mutate(BaseToken = fct_reorder(BaseToken, RankIdx),
         TokenRankIdx = case_when((Ordered=='True')|(Binary=='True')~OrderIdx,
                                  T~(TokenRankIdx-1)))

color.scale.calc.df <- types.timeSHAP.df %>%
  select(BaseToken,Token,Binary,Ordered,TokenRankIdx) %>%
  unique() %>%
  group_by(BaseToken) %>%
  mutate(MaxOrderIdx = max(TokenRankIdx,na.rm = T)) %>%
  ungroup() %>%
  mutate(ColorScale = case_when(MaxOrderIdx>0~(TokenRankIdx)/(MaxOrderIdx),
                                T~1))

types.timeSHAP.df <- types.timeSHAP.df %>%
  left_join(color.scale.calc.df)

# Create feature importance beeswarm plot for static predictors of each type
static.Imaging.timeshap.plot <- types.timeSHAP.df %>%
  filter(Baseline=='Static',
         Type == 'Brain Imaging') %>%
  types.timeSHAP.plots()

static.DemoSES.timeshap.plot <- types.timeSHAP.df %>%
  filter(Baseline=='Static',
         Type == 'Demographics and Socioeconomic Status',
         abs(SHAP)<=.35) %>%
  types.timeSHAP.plots()

static.ERCare.timeshap.plot <- types.timeSHAP.df %>%
  filter(Baseline=='Static',
         Type == 'Emergency Care and ICU Admission',
         abs(SHAP)<=.5) %>%
  types.timeSHAP.plots()

static.ICUMedsMx.timeshap.plot <- types.timeSHAP.df %>%
  filter(Baseline=='Static',
         Type == 'ICU Medications and Management') %>%
  types.timeSHAP.plots()

static.Injury.timeshap.plot <- types.timeSHAP.df %>%
  filter(Baseline=='Static',
         Type == 'Injury Characteristics and Severity',
         abs(SHAP)<=.1) %>%
  types.timeSHAP.plots()

static.Labs.timeshap.plot <- types.timeSHAP.df %>%
  filter(Baseline=='Static',
         Type == 'Labs',
         abs(SHAP)<=.2) %>%
  types.timeSHAP.plots()

static.MedBehavHx.timeshap.plot <- types.timeSHAP.df %>%
  filter(Baseline=='Static',
         Type == 'Medical and Behavioural History',
         abs(SHAP)<=.1) %>%
  types.timeSHAP.plots()

static.SurgMonitor.timeshap.plot <- types.timeSHAP.df %>%
  filter(Baseline=='Static',
         Type == 'Surgery and Neuromonitoring') %>%
  types.timeSHAP.plots()

# Create feature importance beeswarm plot for dynamic predictors of each type
dynamic.Imaging.timeshap.plot <- types.timeSHAP.df %>%
  filter(Baseline=='Dynamic',
         Type == 'Brain Imaging',
         abs(SHAP)<=.025) %>%
  types.timeSHAP.plots()

dynamic.ICUMedsMx.timeshap.plot <- types.timeSHAP.df %>%
  filter(Baseline=='Dynamic',
         Type == 'ICU Medications and Management',
         abs(SHAP)<=.15) %>%
  types.timeSHAP.plots()

dynamic.ICUVitals.timeshap.plot <- types.timeSHAP.df %>%
  filter(Baseline=='Dynamic',
         Type == 'End-of-day Assessment',
         abs(SHAP)<=.2) %>%
  types.timeSHAP.plots()

dynamic.Labs.timeshap.plot <- types.timeSHAP.df %>%
  filter(Baseline=='Dynamic',
         Type == 'Labs',
         abs(SHAP)<=.025) %>%
  types.timeSHAP.plots()

dynamic.SurgMonitor.timeshap.plot <- types.timeSHAP.df %>%
  filter(Baseline=='Dynamic',
         Type == 'Surgery and Neuromonitoring',
         abs(SHAP)<=.2) %>%
  types.timeSHAP.plots()