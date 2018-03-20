rm(list=ls())

path_RData <- paste("/Users/baifrank/Desktop/recomm_output/ml_1m_output/",'matrix_original.RData',sep='')
load(path_RData)

##计算相关函数定义部分
mat_input <- mat_train
##构造一个矩阵来储存相关系数矩阵
mat_sim <- matrix(0,nrow=nrow(mat_input),ncol=nrow(mat_input))
system.time(
  for (i in 1:(nrow(mat_input)-1)){
    print(paste("Calculating:User-",i,"/",nrow(mat_input),sep=''))
    for (j in (i+1):nrow(mat_input)){
      sim <- cor(mat_input[i,],mat_input[j,],use="pairwise.complete.obs")
      mat_sim[i,j] <- sim
    }
  }
)

correlation_KNN <- 0
mat_sim_KNN <- mat_sim
mat_sim_KNN[which(mat_sim_KNN < correlation_KNN)] <- 0
mat_sim_KNN[which(is.na(mat_sim_KNN)==T)] <- 0
  

##将相似度矩阵填满
mat_sim_KNN_fill <- mat_sim_KNN
mat_sim_KNN_fill <- mat_sim_KNN_fill+t(mat_sim_KNN_fill)

##定义评分标准化的函数
##将每个用户的评分减去平均分-->去除每个用户的打分偏好
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
      ##如果所有用户均未对该物品打分，那么直接用最大可能性打分，否则使用最近邻公式计算
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


##--main程序--##
##预测打分
mat_pre <- predict_UBCF(mat_train,mat_sim_KNN_fill)
##计算误差
compute_error(mat_test,mat_pre)
