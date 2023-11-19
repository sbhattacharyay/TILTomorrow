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
