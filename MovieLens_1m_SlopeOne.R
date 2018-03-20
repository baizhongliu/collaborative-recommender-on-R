##Slope One预测器
rm(list=ls())

library(dplyr)
library(reshape2)
path_RData <- paste("/Users/baifrank/Desktop/recomm_output/ml_1m_output/",'matrix_original.RData',sep='')
load(path_RData)

##构造Slope One预测器
##构造储存预测打分的矩阵
mat_pre <- matrix(0,nrow=nrow(mat_train),ncol=ncol(mat_train))
##Alternative:如果不将权重列入计算中，最终只是将平均值作为预测值
##为了节约计算成本，这里只提取test集中评价的物品
system.time(
  for(i in 1:nrow(mat_train)){
    print(paste("Calculating:",i,'/',nrow(mat_train),sep=''))
    rating_user <- mat_train[i,]
    rating_user_test <- mat_test[i,]
    ##把该用户在test集中评价的电影index提取出来
    index_no_rating <- which(is.na(rating_user_test)==F)
    len_no_rating <- length(index_no_rating)
    for(j in 1:len_no_rating){
      print(paste("Calculating:",i,'/',nrow(mat_train),' ',j,'/',len_no_rating,sep=''))
      ##该未打分物品与其它物品的距离矩阵
      index <- index_no_rating[j]
      mat_distance <- mat_train[,index]-mat_train
      ##计算平均距离
      distance_mean <- apply(mat_distance[-i,],2,mean,na.rm=T)
      ##偏移加上物品原本的评分
      rating_add <- distance_mean+rating_user
      ##预测评分
      mat_pre[i,index] <- mean(rating_add[-index],na.rm=T)
    }
  }
)


##定义计算误差的函数
compute_error <- function(mat_real,mat_pre){
  error_list <- list()
  ##四舍五入到整数(5以上赋值为5，0以下赋值为0)
  mat_pre_int <- mat_pre
  mat_pre_int[which(mat_pre_int>5)] <-5
  mat_pre_int[which(mat_pre_int<0)] <-0
  mat_pre_int <- round(mat_pre_int)
  
  ##计算误差
  n_test <- length(which(is.na(mat_real)==F))
  mat_abs_error <- abs(mat_real-mat_pre_int)
  MAE_UBCF <- sum(mat_abs_error,na.rm=T)/n_test
  MSE_UBCF <- sum(mat_abs_error^2,na.rm=T)/n_test
  RMSE_UBCF <- sqrt(MSE_UBCF)
  
  error_list$MAE_UBCF <- MAE_UBCF
  error_list$MSE_UBCF <- MSE_UBCF
  error_list$RMSE_UBCF <- RMSE_UBCF
  return(error_list)
}

compute_error(mat_test,mat_pre)



