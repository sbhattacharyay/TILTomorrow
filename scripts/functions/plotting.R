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
    coord_cartesian(ylim = c(0.23,1)) +
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
    coord_cartesian(ylim = c(0.5423,1.59)) +
    xlab("Day of ICU stay")+
    ylab("Calibration slope")+
    ggtitle(title)+
    scale_x_discrete(expand = expansion(mult = c(.05, .05)))+
    scale_y_continuous(breaks = seq(0.6,1.5,.2), expand = c(0,0)) +
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
  non_consec_wis = [window.indices[i] for i in (np.where(np.diff(window.indices) != 1)[0]+1)]
  
  # Iterate through non-consecutive windows
  for curr_idx in non_consec_wis:
    
    # Identify GUPIs with missing true label at current non-consecutive window
    curr_missing_GUPIs = pred_df[(pred_df.WindowIdx==curr_idx)&(pred_df.TrueLabel.isna())].GUPI.unique()
  
  # Identify instances in which the consecutive window index has a non-missing true label for the current missing GUPI set
  replacements = pred_df[pred_df.GUPI.isin(curr_missing_GUPIs) & (pred_df.WindowIdx.isin([curr_idx-1,curr_idx+1])) & (pred_df.TrueLabel.notna())].reset_index(drop=True)
  
  # If there are viable consecutive window indices, replace missing values with them
  if replacements.shape[0] != 0:
    
    # Use the highest window index if others are available
    replacements = replacements.loc[replacements.groupby(['GUPI','TUNE_IDX','REPEAT','FOLD','SET']).WindowIdx.idxmax()].reset_index(drop=True)
  
  # Identify which rows shall be replaced 
  remove_rows = replacements[['GUPI','TUNE_IDX','REPEAT','FOLD','SET']]
  remove_rows['WindowIdx'] = curr_idx
  
  # Add indicator designating rows for replacement
  pred_df = pred_df.merge(remove_rows,how='left',indicator=True)
  
  # Rectify window index in replacement dataframe
  replacements['WindowIdx'] = curr_idx
  
  # Replace rows with missing true label with viable, consecutive-window replacement
  pred_df = pd.concat([pred_df[pred_df._merge!='both'].drop(columns='_merge'),replacements],ignore_index=True).sort_values(by=['REPEAT','FOLD','TUNE_IDX','GUPI']).reset_index(drop=True)
  
  else:
    pass
  
  # Filter dataframe to desired window indices
  pred_df = pred_df[pred_df.WindowIdx.isin(window.indices)].reset_index(drop=True)
  
  # Return filtered dataframe
  return(pred_df)

}
