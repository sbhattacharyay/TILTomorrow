# Function to plot TILBasic-specific feature-level TimeSHAP values
TILBasic.timeSHAP.plots <- function(plot.df){
  curr.timeSHAP.plot <- plot.df %>% 
    ggplot() +
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
      axis.text.y = element_text(size = 6, color = 'black'),
      axis.title.x = element_blank(),
      panel.border = element_blank(),
      axis.line.x = element_line(size=1/.pt),
      axis.text = element_text(color='black'),
      legend.position = 'none',
      panel.grid.major.y = element_blank(),
      panel.spacing = unit(10, 'points'),
      plot.margin=grid::unit(c(0,2,0,0), "mm")
    )
  return(curr.timeSHAP.plot)
}

# Function to plot relevance layer boxplots
relevance.boxplots <- function(plot.df){
  curr.boxplot <- plot.df %>%
    mutate(
      GROUPS = fct_relevel(GROUPS, 'Top', 'Middle', 'Bottom'),
      BaseToken = fct_reorder(BaseToken, median)
    ) %>% 
    ggplot(aes(y = BaseToken,fill=Type)) +
    geom_boxplot(aes(xmin=min,xmax=max,xlower=Q1,xupper=Q3,xmiddle=median),stat='identity') +
    facet_grid(rows = vars(GROUPS), scales = 'free_y', switch = 'y', space = 'free_y') +
    scale_x_continuous(expand = c(0, 0.1)) +
    scale_y_discrete(expand = c(0,0)) + 
    #scale_fill_manual(values=c("#003f5c", "#444e86", "#955196",'#dd5182','#ff6e54','#ffa600')) +
    xlab('Learned relevance weight')+
    theme_minimal(base_family = 'Roboto Condensed') +
    theme(
      strip.background = element_blank(),
      strip.text.y = element_blank(),
      axis.title.y = element_blank(),
      axis.text.x = element_text(size = 10, color = 'black'),
      axis.text.y = element_text(size = 10, color = 'black',face = 'bold'),
      axis.title.x = element_text(size = 12, color = 'black',face = 'bold'),
      panel.border = element_blank(),
      axis.line.x = element_line(size=1/.pt),
      axis.text = element_text(color='black'),
      legend.position = 'bottom',
      panel.grid.major.y = element_blank(),
      panel.spacing = unit(10, 'points'),
      legend.key.size = unit(1.3/.pt,'line'),
      legend.title = element_text(size = 12, color = 'black',face = 'bold'),
      legend.text=element_text(size=10)
    )
  return(curr.boxplot)
}

# Function to plot pre-post TILBasic distributions
pre.post.TILBasic.dist.plots <- function(plot.df,title){
  curr.dist.plot <- plot.df %>%
    ggplot(aes(fill=fct_rev(postTILBasic), y=pct, x=ICUDay)) + 
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
    ggtitle(title)+
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
      # legend.position = 'bottom',
      # legend.title = element_text(size = 7, color = "black", face = 'bold'),
      # legend.text=element_text(size=6),
      # legend.key.size = unit(1.3/.pt,"line")
    )
  return(curr.dist.plot)
}

# Function to plot threshold-level AUCs
thresh.level.AUC.plot <- function(plot.df,title,color.palette){
  curr.AUC.plot <- ggplot() +
    geom_hline(yintercept = .5,alpha=1,linetype = "dashed",size=1.75/.pt, color = 'gray') +
    geom_ribbon(data=plot.df %>% filter(Grouping==1),
                mapping=aes(x=ICUDay, ymin=lo, ymax=hi, fill=VariableSet, group = VariableSet),
                alpha=.2) +
    geom_line(data=plot.df %>% filter(Grouping==1),
              mapping=aes(x=ICUDay, y=median, color=VariableSet, group = VariableSet),
              lwd=1.75/.pt) +
    geom_errorbar(data=plot.df %>% filter(Grouping!=1),
                  mapping=aes(x=ICUDay, ymin=lo, ymax=hi, color=VariableSet),
                  position = position_dodge(width = .75),
                  width=.35) +
    geom_point(data=plot.df %>% filter(Grouping!=1),
               mapping=aes(x=ICUDay, y=median, color=VariableSet),
               position = position_dodge(width = .75),
               size=.75) +
    facet_grid(cols = vars(Grouping), scales = 'free_x', switch = 'x', space = 'free_x') +
    coord_cartesian(ylim = c(0.4363810,1)) +
    scale_fill_manual(values = color.palette)+
    scale_color_manual(values = color.palette)+
    xlab("Day of ICU stay")+
    ylab("Area under ROC curve (AUC)")+
    ggtitle(title)+
    scale_x_discrete(expand = expansion(mult = c(.05, .05)))+
    scale_y_continuous(breaks = seq(0.3,1,.1), expand = c(0,0)) +
    theme_minimal(base_family = 'Roboto Condensed') +
    theme(
      axis.text.x = element_text(size = 6, color = "black",margin = margin(0,0,0,0)),
      axis.text.y = element_text(size = 5, color = "black",margin = margin(0,0,0,0)),
      axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
      #axis.title.y = element_text(size = 7, color = "black",face = 'bold',margin = margin(0,0,0,0)),
      axis.title.y = element_blank(),
      panel.border = element_blank(),
      axis.line.x = element_line(size=1/.pt),
      axis.text = element_text(color='black'),
      legend.position = 'none',
      panel.spacing = unit(5, 'points'),
      plot.margin=grid::unit(c(0,2,0,0), "mm"),
      strip.text = element_blank(),
      plot.title = element_text(size=7, color = "black",face = 'bold',margin = margin(b = .5),hjust = .5)
    )
  return(curr.AUC.plot)
}

# Function to plot threshold-level calibration slopes
thresh.level.calib.slope.plot <- function(plot.df,title){
  curr.calib.slope.plot <- ggplot() +
    geom_hline(yintercept = 1, color='#ffa600',alpha = 1, size=1.75/.pt)+
    geom_ribbon(data=plot.df %>% filter(Grouping==1),
                mapping=aes(x=ICUDay, ymin=lo, ymax=hi, group=1),
                fill='#003f5c',
                alpha=.2) +
    geom_line(data=plot.df %>% filter(Grouping==1),
              mapping=aes(x=ICUDay, y=median, group=1),
              color='#003f5c',
              lwd=1.75/.pt) +
    geom_errorbar(data=plot.df %>% filter(Grouping!=1),
                  mapping=aes(x=ICUDay, ymin=lo, ymax=hi, group=1),
                  color='#003f5c',
                  position = position_dodge(width = .75),
                  width=.35) +
    geom_point(data=plot.df %>% filter(Grouping!=1),
               mapping=aes(x=ICUDay, y=median, group=1),
               color='#003f5c',
               position = position_dodge(width = .75),
               size=.75) +
    facet_grid(cols = vars(Grouping), scales = 'free_x', switch = 'x', space = 'free_x') +
    coord_cartesian(ylim = c(0,1.59)) +
    xlab("Day of ICU stay")+
    ylab("Calibration slope")+
    ggtitle(title)+
    scale_x_discrete(expand = expansion(mult = c(.05, .05)))+
    scale_y_continuous(breaks = seq(0,1.5,.5), expand = c(0,0)) +
    theme_minimal(base_family = 'Roboto Condensed') +
    theme(
      axis.text.x = element_text(size = 6, color = "black",margin = margin(0,0,0,0)),
      axis.text.y = element_text(size = 5, color = "black",margin = margin(0,0,0,0)),
      axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
      axis.title.y = element_text(size = 7, color = "black",face = 'bold',margin = margin(0,0,0,0)),
      # axis.title.y = element_blank(),
      panel.border = element_blank(),
      axis.line.x = element_line(size=1/.pt),
      axis.text = element_text(color='black'),
      legend.position = 'none',
      panel.spacing = unit(5, 'points'),
      plot.margin=grid::unit(c(0,2,0,0), "mm"),
      strip.text = element_blank(),
      plot.title = element_text(size=7, color = "black",face = 'bold',margin = margin(b = .5),hjust = .5)
    )
  return(curr.calib.slope.plot)
}

# Function to prepare formatted TIL dataframe for plotting
prepare.df <- function(TIL.df,window.indices){
  
  # Determine non-consecutive window indices
  non.consec.wis <- window.indices[c(FALSE,diff(window.indices) != 1)]
  
  # Iterate through non-consecutive windows
  for (curr.idx in non.consec.wis){
    
    # Identify GUPIs with missing true label at current non-consecutive window
    curr.missing.GUPIs <- TIL.df %>%
      filter(TILTimepoint == curr.idx,
             is.na(TILBasic)) %>%
      .$GUPI %>%
      unique()
    
    # Identify instances in which the consecutive window index has a non-missing true label for the current missing GUPI set
    replacements <- TIL.df %>%
      filter(GUPI %in% curr.missing.GUPIs,
             TILTimepoint %in% c(curr.idx-1,curr.idx+1),
             !is.na(TILBasic))
    
    # If there are viable consecutive window indices, replace missing values with them
    if (dim(replacements)[1] != 0){
      
      # Use the highest window index if others are available
      replacements <- replacements %>%
        group_by(GUPI) %>%
        slice_max(TILTimepoint) %>%
        mutate(TILTimepoint = curr.idx)
      
      # Replace rows with missing true label with viable, consecutive-window replacement
      TIL.df <- TIL.df %>%
        anti_join(replacements %>%
                    select(GUPI,TILTimepoint,WindowTotal)) %>%
        rbind(replacements) %>%
        arrange(GUPI,TILTimepoint)
    }
  }
  # Filter dataframe to desired window indices
  TIL.df <- TIL.df %>%
    filter(TILTimepoint %in% window.indices)
  
  # Return filtered dataframe
  return(TIL.df)
}

# Store study window and TILBasic extraction script into function
get.formatted.TILBasic <- function(focus.timepoints){
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
    prepare.df(focus.timepoints) %>%
    mutate(ICUDay = sprintf('Day %.0f',TILTimepoint),
           ICUDay = fct_reorder(factor(ICUDay), TILTimepoint),
           TILBasic = case_when(TILBasic==4~'4',
                                TILBasic==3~'3',
                                TILBasic==2~'2',
                                TILBasic==1~'1',
                                TILBasic==0~'0',
                                is.na(TILBasic)~'Missing'))
  
  # Load patient ICU admission/discharge timestamps
  CENTER.TBI.ICU.discharge.info <- read.csv('../../center_tbi/CENTER-TBI/adm_disch_timestamps.csv',na.strings = c("NA","NaN","", " ")) %>%
    filter(GUPI %in% study.days.TILBasic$GUPI) %>%
    select(GUPI,ICUDischTimeStamp) %>%
    mutate(ICUDischTimeStamp = as.POSIXct(substr(ICUDischTimeStamp,1,10),format = '%Y-%m-%d',tz = 'GMT')) %>%
    arrange(GUPI,ICUDischTimeStamp)
  
  # Load patient withdrawal-of-life-sustaining-therapies (WLST) information
  CENTER.TBI.WLST.info <- read.csv('../../center_tbi/CENTER-TBI/WLST_patients.csv',na.strings = c("NA","NaN","", " ")) %>%
    filter(GUPI %in% study.days.TILBasic$GUPI) %>%
    select(GUPI,ends_with('TimeStamp')) %>%
    pivot_longer(cols=-c(GUPI),names_to = 'TypeOfTimeStamp',values_to = 'WLSTTimeStamp') %>%
    mutate(WLSTTimeStamp = as.POSIXct(substr(WLSTTimeStamp,1,10),format = '%Y-%m-%d',tz = 'GMT')) %>%
    filter(TypeOfTimeStamp != 'ICUDischTimeStamp',
           !is.na(WLSTTimeStamp)) %>%
    group_by(GUPI) %>%
    slice_min(WLSTTimeStamp,with_ties=F,n=1) %>%
    arrange(GUPI,WLSTTimeStamp)
  
  # Load patient death information
  CENTER.TBI.death.info <- read.csv('../../center_tbi/CENTER-TBI/death_patients.csv',na.strings = c("NA","NaN","", " ")) %>%
    filter(GUPI %in% study.days.TILBasic$GUPI) %>%
    select(GUPI,ICUDischargeStatus,ends_with('TimeStamp')) %>%
    pivot_longer(cols=-c(GUPI,ICUDischargeStatus),names_to = 'TypeOfTimeStamp',values_to = 'DeathTimeStamp') %>%
    mutate(DeathTimeStamp = as.POSIXct(substr(DeathTimeStamp,1,10),format = '%Y-%m-%d',tz = 'GMT')) %>%
    filter(TypeOfTimeStamp != 'ICUDischTimeStamp',
           !is.na(DeathTimeStamp)) %>%
    group_by(GUPI) %>%
    slice_min(DeathTimeStamp,with_ties=F,n=1) %>%
    arrange(GUPI,DeathTimeStamp)
  
  # Merge discharge, WLST, and death timestamps into single dataframe
  CENTER.TBI.ICU.timestamps <- CENTER.TBI.ICU.discharge.info %>%
    left_join(CENTER.TBI.WLST.info %>% select(GUPI,WLSTTimeStamp)) %>%
    left_join(CENTER.TBI.death.info %>% select(GUPI,DeathTimeStamp)) %>%
    left_join(study.days.TILBasic %>%
                mutate(BaseDate = TimeStampStart - as.difftime(TILTimepoint, unit="days")) %>%
                select(GUPI,BaseDate) %>%
                drop_na(BaseDate) %>%
                unique()) %>%
    filter(!(is.na(WLSTTimeStamp) & is.na(DeathTimeStamp))) %>%
    mutate(WLSTTimepoint = as.numeric(WLSTTimeStamp - BaseDate,units="days"),
           DeathTimepoint = as.numeric(DeathTimeStamp - BaseDate,units="days"),
           WLSTOrDeathTimepoint = case_when(WLSTTimepoint>DeathTimepoint ~ DeathTimepoint,
                                            WLSTTimepoint<DeathTimepoint ~ WLSTTimepoint,
                                            T ~ DeathTimepoint)) %>%
    filter(WLSTOrDeathTimepoint <= max(focus.timepoints))
  
  # Create an empty vector to store Death/WLST rows
  death.WLST.list <- vector(mode = "list")
  i <- 0
  
  # Iterate through rows of death or WLST to create dataframe rows
  for (curr.GUPI in unique(CENTER.TBI.ICU.timestamps$GUPI)){
    
    # Add one to running row index
    i <- i + 1
    
    # Filter current death/WLST information
    curr.GUPI.timestamps <- CENTER.TBI.ICU.timestamps %>%
      filter(GUPI == curr.GUPI)
    
    # Create rows corresponding to death/WLST for study window dataframe
    curr.GUPI.rows <- data.frame(GUPI = curr.GUPI,
                                 TILTimepoint = seq(min(curr.GUPI.timestamps$WLSTOrDeathTimepoint),max(focus.timepoints)),
                                 TILBasicMarker = 'WLST or Died') %>%
      filter(TILTimepoint %in% focus.timepoints) %>%
      mutate(ICUDay = sprintf('Day %.0f',TILTimepoint))
    
    # Append current rows corresponding to death/WLST to running list
    death.WLST.list[[i]] <- curr.GUPI.rows
  }
  
  # Concatenate list of death/WLST rows into single dataframe
  study.days.and.WLST.death.TILBasic <- study.days.TILBasic %>%
    full_join(do.call(rbind,death.WLST.list)) %>%
    mutate(TILBasic = case_when(!is.na(TILBasic) ~ TILBasic,
                                T ~ TILBasicMarker)) %>%
    select(-TILBasicMarker) %>%
    arrange(GUPI,TILTimepoint)
  
  # Create grid of days which are unaccounted for to mark as discharged
  discharge.days <- expand_grid(GUPI=unique(study.days.and.WLST.death.TILBasic$GUPI),TILTimepoint=focus.timepoints) %>%
    anti_join(study.days.and.WLST.death.TILBasic) %>%
    mutate(TILBasic = 'Discharged',
           ICUDay = sprintf('Day %.0f',TILTimepoint))
  
  # Join the discharged timepoints to study window TILBasic dataframe
  study.days.WLST.death.and.discharge.TILBasic <- study.days.and.WLST.death.TILBasic %>%
    full_join(discharge.days) %>%
    arrange(GUPI,TILTimepoint) %>%
    select(GUPI,ICUDay,TILTimepoint,TILBasic)
  
  # Return final formatted study window dataframe
  return(study.days.WLST.death.and.discharge.TILBasic)
}