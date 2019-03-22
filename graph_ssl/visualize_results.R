library(tibble)
library(dplyr)
library(ggplot2)

setwd("/home/klaus/eclipse-workspace/NN2018/graph_ssl")
DF_FOLDER = "./results/joined/"

dfCombine <- function(df_list){
  #Fix missing columns by introducing NA
  all_colnames = unique(unlist(sapply(df_list,
                                      colnames)))
  for (i in 1:length(df_list)) {
    for (j in all_colnames) {
      if (!j %in% colnames((df_list[[i]]))){
        df_list[[i]] <-  cbind(df_list[[i]],rep(NA,nrow(df_list[[i]])))
        colnames(df_list[[i]])[ncol(df_list[[i]])] <- j
      }
    }
  }
  
  df <- as.tibble(do.call(rbind,df_list))
  return(df)
}
dfRead <- function(fPath) {
  df <- read.csv(fPath,sep=",")
  df <- df[,colnames(df) != "X"]
  if ("dataset_sd" %in% colnames(df)){
    df$dataset <- paste(df$dataset,"_sd=",df$dataset_sd,sep="")
    df <- df[,colnames(df) != "dataset_sd"]
  }
  return(df)
}

df_gaussian  <-dfRead(paste0(DF_FOLDER,"joined_gaussian_dynamic_v2.csv"))

df_spiral_fixed  <- dfCombine(list(
  dfRead(paste0(DF_FOLDER,"joined_spiral_fixed_v2.csv")),
  dfRead(paste0(DF_FOLDER,"joined_spiral_fixed_v2_LP.csv"))
  ))

df_spiral_dynamic <- dfRead(paste0(DF_FOLDER,"joined_spiral_recalculated_2.csv"))


#df_spiral <- df_spiral[,colnames(df_spiral) != "X"]
#df_spiral$dataset <- paste(df_spiral$dataset,"_sd=",df_spiral$dataset_sd)



df_list = list(df_gaussian,df_spiral_dynamic)
df = dfCombine(df_list)



#Round off values
df[,sapply(df,class)=="numeric"] <-  round(df[,sapply(df,class)=="numeric"],digits=5)
output_variables <- c("experiments","mean_acc","sd_acc","elapsed_time",
                      "min_acc","max_acc","median_acc","each_acc")
affmat_variables <- colnames(df)[sapply(colnames(df),
                                        function(x){grepl("aff_",x)})]
hyperparams <- !(colnames(df) %in% c(affmat_variables,output_variables))
hyperparams <- colnames(df)[hyperparams]




#######################################################################
EXP_NAME = "gaussians/dynamic/LDST"
for (ds in unique(df$dataset)) {
for (lp in c(0.1)){
#########Adicione colunas extra######################
df <- df[,all_colnames]
#df$xtremeAlpha <- df$algorithm != "LGC" | factor(df$alpha) %in% c(0.9990,1e-04) 
df$cond1 <-   (df$algorithm %in% c("LGC","LP","RF"))
  df$isSpiral <- grepl(pattern = "spiral",df$dataset)

filterConfig <- df[1,hyperparams]
########Suas configs aqui############################
filterConfig$labeled_percent <- lp
filterConfig$dataset <- ds
filterConfig$algorithm <- NA
#filterConfig$xtremeAlpha <- NA
filterConfig$corruption_level <- NA
filterConfig$tuning_iter <- NA
filterConfig$isSpiral <- F
filterConfig$cond1 <- F
filterConfig$alpha <- NA


filter <- function(df,filterConfig,discarded_vars=c()) {
  filterConfig<- filterConfig[,!colnames(filterConfig)%in%discarded_vars,]
  filterConfig <- filterConfig[,
                               !as.logical(is.na(filterConfig))]
  print(filterConfig)
  filter_bool <- sapply(1:nrow(df),function(i){
    for (x in colnames(filterConfig)){
      nacomp_xor <- xor(is.na(filterConfig[[1,x]]),
                    is.na(df[[i,x]]))
      nacomp_and <- is.na(filterConfig[[1,x]]) &&
                        is.na(df[[i,x]])
      if(is.na(df[[i,x]]) ){
        next
      }
      
      if (nacomp_xor) {
        return(FALSE)
      }
      
      if (!nacomp_and && filterConfig[[1,x]] != df[[i,x]]){
        return(FALSE)
      }
    }
    return(TRUE)
  })
  return((df[filter_bool,]))
}
filtered_df <- filter(df,filterConfig)
if (nrow(filtered_df)==0){next}
####################################################
linePlot <- function(df,x_var,use_sd=T, y_var="mean_acc",
                     ignore_var = c("elapsed_time","sd_acc","mean_acc",
                                    "min_acc","each_acc","max_acc","median_acc")) {
  toStrExceptNA <- function(a,sep=";"){
    temp <- as.logical(!is.na(a[1,]))
    a <- a[1,temp]
    if (length(a)==0) return("")
    return(paste0(sapply(1:length(a),
                         function(i){
                           paste0(colnames(a)[i],"=",
                                  as.character(a[[1,i]]),
                                  sep)
                         }
    ),collapse=""))
    
  }
  compareNA <- function(v1,v2) {
    same <- (v1 == v2) | (is.na(v1) & is.na(v2))
    same[is.na(same)] <- FALSE
    return(same)
  }
  ignore_var <- ignore_var[!ignore_var %in% c(x_var,y_var)] 
  
  b1 <- !apply(df,2,function(x){
    all(sapply(x,function(y)compareNA(x[1],y)))}
    )
  b2 <- !colnames(df) %in% c(x_var,y_var,ignore_var)
  b2 <- !colnames(df) %in% c(x_var,y_var,ignore_var)
  g <- sapply(1:nrow(df[,b1 & b2]),
              function(i){
                return(toStrExceptNA(df[i,b1 & b2]))
              }
  )
  
  g_compl <- sapply(1:nrow(df[,!b1]),
                    function(i){
                      return(toStrExceptNA(df[i,!b1],sep=";\n"))
                    }
  )
  print(g)
  
  n <- nrow(df)
  new_df <- as.tibble(data.frame(setting=factor(n),x=numeric(n),y=numeric(n)))
  new_df$setting <- as.factor(g)
  new_df$x <- sapply(df[,x_var],as.numeric)
  new_df$y <- sapply(df[,y_var],as.numeric)
  new_df$y_min <- sapply(df[,y_var] - df[,"sd_acc"],as.numeric)
  new_df$y_max <- sapply(df[,y_var] + df[,"sd_acc"],as.numeric)
  
  
  g <- ggplot(data = new_df,aes(x,y,
              ymin =y_min, ymax = y_max,
              group=setting,colour=setting,fill=setting)) +
    geom_line()
  if(use_sd){
    g <- g + geom_ribbon(alpha=0.5) 
  }
  g <- g + scale_color_brewer(palette="Dark2") +
    scale_fill_brewer(palette="Dark2") +
    scale_y_continuous(breaks = seq(0.5,1,0.1)) +
    ggtitle(paste0(g_compl[1])) + xlab(as.character(x_var)) +
    ylab(as.character(y_var)) 
  return(g)
}

########################
export_folder = file.path(getwd(),"plots_R",EXP_NAME,
                         paste0('dataset=',as.character(ds)),
                         paste0('labeledPercent=',as.character(lp))
                         ) 
if (!dir.exists(export_folder)){
  dir.create(export_folder,recursive = T)
}

g <- linePlot(filtered_df,"corruption_level",use_sd = F)
plot(g)
print(file.path(export_folder,'mean_acc.png'))
ggsave(file.path(export_folder,'mean_acc.png'), width = 12, height = 8,dpi=150)

g <- linePlot(filtered_df,"corruption_level",use_sd = T)
ggsave(file.path(export_folder,'mean_acc_with_sd.png'), width = 12, height = 8,dpi=150)

if ("median_acc" %in% colnames(filtered_df) &
    any(!is.na(filtered_df$median_acc))) {
  g <- linePlot(filtered_df[!is.na(filtered_df$median_acc),],
                "corruption_level",y_var="median_acc",use_sd = F)
  ggsave(file.path(export_folder,'median_acc.png'), width = 12, height = 8,dpi=150)
}

}
}
######################################################
ggsave('./plots_R/dataset=spiral;labeled_percent=0.5;mean_acc.png', width = 12, height = 8,dpi=150)
###############################################################
library(mlbench)
plot(mlbench::mlbench.spirals(1500,cycles=1,sd=0.08))
library(SSLKlaus)
X = data.table(read.csv("spiral_hard_X.csv")[,2:3])
colnames(X) = c("x1","x2")
X[,"y"] = read.csv("spiral_hard_Y.csv")[,2]
ggplot(X,aes(x1,x2,color=y)) + geom_point()

for (sd in c(0.4,1,2,3))
{
temp = mlbench::mlbench.2dnormals(n=1000,sd = sd)
X = as.tibble(temp$x)
colnames(X) = c("V1","V2")
write.csv(X,
          paste0("gaussians_sd=",sd,"_X.csv"))
X[,"y"] = temp$classes
ggplot(X,aes(V1,V2,color=y)) + geom_point()

write.csv(as.numeric(temp$classes),
          paste0("gaussians_sd=",sd,"_Y.csv"))
}