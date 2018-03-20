##MovieLens_1m
##矩阵分解SVD
##如果用户对电影没有评分，则赋值0
##通过潜在因子计算相似度
##得到关于用户和电影的潜在因子之后使用线性回归/随机森林回归树
library(dplyr)
library(reshape2)
rm(list=ls())

path_RData <- paste("/Users/baifrank/Desktop/recomm_output/ml_1m_output/",'matrix_original.RData',sep='')
load(path_RData)

##对矩阵NA值填充0值之后进行SVD分解
mat_fill_0 <- mat_train
mat_fill_0[which(is.na(mat_fill_0)==T)] <- 0
system.time(mat_svd <- svd(mat_fill_0))

##user和movie id
user_id <- as.numeric(rownames(mat_train))
movie_id <- as.numeric(colnames(mat_train))

##定义输出user、movie隐因子data.frame的函数
extract_factor <- function(result_svd,n_user_fac,n_movie_fac){
  ##构造list用来分别储存user和movie的latent factor
  list_factor <- list()
  
  ##指定user和movie的隐因子数量:如果为0则代表不提取该部分的隐因子
  if(n_user_fac==0){
    list_factor$user <- NULL
  }else{
    user_factor <- result_svd$u[,1:n_user_fac]
    user_factor_df <- data.frame(cbind(user_id,user_factor))
    colnames(user_factor_df) <- c("user_id",paste("user_factor",1:n_user_fac,sep=''))
    list_factor$user <- user_factor_df
  }
  
  if(n_movie_fac==0){
    list_factor$movie <- NULL
  }else{
    movie_factor <- result_svd$v[,1:n_movie_fac]
    movie_factor_df <- data.frame(cbind(movie_id,movie_factor))
    colnames(movie_factor_df) <- c("item_id",paste("item_factor",1:n_movie_fac,sep=''))
    list_factor$movie <- movie_factor_df
  }
  return(list_factor)
}

##定义计算误差的函数－－针对vector
compute_error_vector <- function(vec_real,vec_pre){
  list_error <- list()
  vec_pre_int <- vec_pre
  vec_pre_int[which(vec_pre_int>5)] <- 5
  vec_pre_int[which(vec_pre_int<0)] <- 0
  vec_pre_int <- round(vec_pre_int)
  
  list_error$MAE <- sum(abs(vec_real-vec_pre_int))/length(vec_real)
  list_error$MSE <- sum((vec_real-vec_pre_int)^2)/length(vec_real)
  list_error$RMSE <- sqrt(list_error$MSE)
  return(list_error)
}

##Method2——SVD_KNN
##通过潜在因子来寻找最最近邻用户
user_factor <- extract_factor(mat_svd,10,0)$user
##计算用户之间的Pearson相关系数矩阵
mat_factor <- t(user_factor[,-1])
colnames(mat_factor) <- user_factor$user_id
mat_cor_factor <- cor(mat_factor)
##相关系数的分布情况

##定义将相关系数矩阵中低于阈值的转化为0的函数（>=correlation的定义为近邻用户）
cor_transform <- function(mat_cor,correlation_KNN){
  index_KNN <- which(mat_cor<correlation_KNN)
  mat_cor_KNN <- mat_cor
  mat_cor_KNN[index_KNN] <- 0
  ##对角线上的元素设置为0
  return(mat_cor_KNN-diag(diag(mat_cor_KNN)))
}


##定义评分标准化的函数
rating_scale <- function(mat_input){
  rating_user_mean <- apply(mat_input,1,mean,na.rm=TRUE)
  mat_user_mean <- matrix(rep(rating_user_mean,times=ncol(mat_input)),nrow=nrow(mat_input))
  mat_train_scale <- mat_input-mat_user_mean
  ll <- list()
  ll$mean <- rating_user_mean
  ll$scale <- mat_train_scale
  return(ll)
}

##定义预测打分的函数
predict_UBCF <- function(mat_input,sim_input){
  
  ##将原始评分矩阵标准化，并计算每个用户的平均打分
  l <- rating_scale(mat_input)
  rating_user_mean <- l$mean
  mat_train_scale <- l$scale
  
  ##构建一个储存预测打分的矩阵
  mat_pre <- matrix(0,nrow=nrow(mat_input),ncol=ncol(mat_input))
  rownames(mat_pre) <- rownames(mat_input)
  colnames(mat_pre) <- colnames(mat_input)
  
  ##按照设置好的条件对预测矩阵进行填充
  for(i in 1:nrow(mat_input)){
    print(paste("Caculating：User-",i,"/",nrow(mat_input),sep=''))
    ##提取用户本来的打分向量
    rating_user <- mat_input[i,]
    ##得到需要预测打分的物品的index
    index_no_rating <- which(is.na(rating_user)==T)
    ##提取该用户与其他用户的相关系数向量
    correlation_KNN <- sim_input[i,]
    
    ##对于不同的物品进行打分的预测
    for(j in index_no_rating){
      ##所有用户对于该物品的打分情况
      rating_item_KNN <- mat_train_scale[,j]
      ##将未对该物品打分的用户剔除
      ##如果所有用户均未对该物品打分，那么直接返回0分，否则使用最近邻公式计算
      index_is_rating_KNN <- which(is.na(rating_item_KNN)==F)
      if(length(index_is_rating_KNN)==0){
        mat_pre[i,j] <- 0
      }else{
        rating_KNN <- rating_item_KNN[index_is_rating_KNN]
        sim_KNN <- correlation_KNN[index_is_rating_KNN]
        ##如果所有有打分的用户与该用户的相似度都 < 阈值，那么直接赋值0
        if(sum(sim_KNN)==0){
          mat_pre[i,j] <- 0
        }else{
          value_pre <- rating_user_mean[i]+sum(rating_KNN*sim_KNN)/sum(sim_KNN)
          mat_pre[i,j] <- value_pre
        }
      }
    }
  }
  return(mat_pre)
}


##定义计算误差的函数——矩阵
compute_error_mat <- function(mat_real,mat_pre){
  error_list <- list()
  ##四舍五入到整数(5以上赋值为5，0以下赋值为0)
  mat_pre_int <- mat_pre
  mat_pre_int[which(mat_pre_int>5)] <-5
  mat_pre_int[which(mat_pre_int<0)] <-0
  mat_pre_int <- round(mat_pre_int)
  
  ##计算误差
  n_test <- length(which(is.na(mat_real)==F))
  mat_abs_error <- abs(mat_real-mat_pre_int)
  ##MAE
  MAE_UBCF <- sum(mat_abs_error,na.rm=T)/n_test
  ##MSE
  MSE_UBCF <- sum(mat_abs_error^2,na.rm=T)/n_test
  ##RMSE
  RMSE_UBCF <- sqrt(MSE_UBCF)
  
  error_list$MAE_UBCF <- MAE_UBCF
  error_list$MSE_UBCF <- MSE_UBCF
  error_list$RMSE_UBCF <- RMSE_UBCF
  return(error_list)
}

##计算在不同参数下的误差值
correlation_KNN <- 0
mat_cor_factor_KNN <- cor_transform(mat_cor_factor,correlation_KNN)
mat_pre <- predict_UBCF(mat_train,mat_cor_factor_KNN)
result <- compute_error_mat(mat_test,mat_pre)
result

