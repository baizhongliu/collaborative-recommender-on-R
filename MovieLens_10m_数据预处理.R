##实值评分矩阵:size——69878*10677
rm(list=ls())
##主要工作：限制test集的item在train集中
library(plyr)
library(dplyr)
library(reshape2)
library(data.table)
##对数据进行描述
data_path <- '/Users/baifrank/Desktop/硕士毕业论文/数据集/ml-1m/'
col_names <- c('user_id','item_id','rating','timestamp')
ua.base <- fread(paste(data_path,'ratings.dat',sep=''),header=FALSE)
ua.base <- ua.base[,c(1,3,5,7)]
##user id | item id | rating | timestamp
colnames(ua.base) <- col_names
user_all <- unique(ua.base$user_id)##6040
movie_all <- unique(ua.base$item_id)##3706
sort(unique(ua.base$rating))##只存在整数评分

##make continuous id：both users and movies
user_id_df <- data.frame(original = user_all , user_transform = 1:length(user_all))
item_id_df <- data.frame(original = movie_all , transform = 1:length(movie_all))
#write.csv(user_id_df,file="/Users/baifrank/Desktop/recomm_output/ml_1m_output/uid_transform.csv",row.names=FALSE)
#write.csv(item_id_df,file="/Users/baifrank/Desktop/recomm_output/ml_1m_output/item_transform.csv",row.names=FALSE)

ua.base_transform <- left_join(ua.base,user_id_df,c("user_id" = "original"))
ua.base_transform <- ua.base_transform[,c(1,2,3,5)]
colnames(ua.base_transform) <- c("user_id_original","item_id_original","rating","user_id")
ua.base_transform <- left_join(ua.base_transform,item_id_df,c("item_id_original" = "original"))
ua.base_transform <- plyr::rename(ua.base_transform,c("transform" = "item_id"))
ua.base <- select(ua.base_transform,user_id,item_id,rating)
##乱序排列
ua.base <- mutate(ua.base,sort_factor = rnorm(nrow(ua.base)))
ua.base <- arrange(ua.base,user_id,sort_factor)

##choose 10 ratings for each user,make the test set
##统计每个用户评分的电影数量
ua.base_group <- dplyr::group_by(ua.base,user_id)%>%
  dplyr::summarise(count_ratings = n())
seq_count <- sapply(ua.base_group$count_ratings,function(i) seq(1,i))
seq_count <- unlist(seq_count)
##从每个list中随机抽取10个index
index_test <- which(seq_count <= 10)
ua.train <- ua.base[-index_test,]
ua.test <- ua.base[index_test,]

##计算训练集每个用户的平均分：标准化评分
ua.train_group <- dplyr::group_by(ua.train,user_id)%>%
  dplyr::summarise(rating_avg = mean(rating))
ua.train <- left_join(ua.train,ua.train_group)[,c(1,2,3,5)]
ua.test <- ua.test[,c(1,2,3)]
##输出构造好的1m数据集的训练集和测试集
#write.table(ua.train,file='/Users/baifrank/Desktop/recomm_output/ml_1m_output/ua.train.txt',row.names=FALSE)
#write.table(ua.test,file='/Users/baifrank/Desktop/recomm_output/ml_1m_output/ua.test.txt',row.names=FALSE)

##读取
#ua.train <- read.table("/Users/baifrank/Desktop/recomm_output/ml_1m_output/ua.train.txt",header=T)
#ua.test <- read.table("/Users/baifrank/Desktop/recomm_output/ml_1m_output/ua.test.txt",header=T)

movie_train <- unique(ua.train$item_id)
movie_test <- unique(ua.test$item_id)
movie_union <- union(movie_train,movie_test)
##统一训练集和测试集的movie
##处理方法：手动给train、test集添加上这些movie，打分记为NA
item_add_train <- setdiff(movie_union,movie_train)
ua.train_fill <- rbind(ua.train,data.frame(user_id=1,item_id=item_add_train,rating=NA,rating_avg=NA))
ua.train <- ua.train_fill

item_add_test <- setdiff(movie_union,movie_test)
ua.test_fill <- rbind(ua.test,data.frame(user_id=1,item_id=item_add_test,rating=NA))
ua.test <- ua.test_fill
##实值训练集和测试集的DataFrame处置到此
##定义将dataframe转化成matrix的函数
df_to_mat <- function(df){
  mat <- dcast(df,user_id~item_id)
  uid <- mat$user_id
  iid <- colnames(mat)
  mat <- as.matrix(mat[,-1])
  rownames(mat) <- uid
  colnames(mat) <- iid[-1]
  return(mat)
}
##将训练集转化为矩阵的形式:6040*3706,user*item
mat_train <- df_to_mat(ua.train[,1:3])
mat_test <- df_to_mat(ua.test[,1:3])
##直接用RData把两个Matrix进行保存
path_RData <- paste("/Users/baifrank/Desktop/recomm_output/ml_1m_output/",'matrix_original.RData',sep='')
#save("mat_test","mat_train",file=path_RData)







##另外还要将df中的rating进行标准化：减去对应用户的平均打分，0/1——Binary处理
ua_base_group <- group_by(ua.base,user_id)%>%
  summarise(rating_mean = mean(rating))
ua_base_group <- as.data.frame(ua_base_group)
ua_base_scale <- left_join(ua.base,ua_base_group)
ua_base_scale <- mutate(ua_base_scale,rating_binary=as.numeric(rating>=rating_mean))
##>=3的定义为1，否则为0
ua_base_scale$rating_binary_3 <- 0
index_gt_3 <- which(ua_base_scale$rating>=3)
ua_base_scale$rating_binary_3[index_gt_3] <- 1
path_binary <- '/Users/baifrank/Desktop/recomm_output/MovieLens_10M_output/data_binary/'
path_binary_train <- paste(path_binary,'ua_base_df_continuous_binary.txt')
#write.table(ua_base_scale,file=path_binary_train,row.names=FALSE)
ua_test_scale <- left_join(ua.test,ua_base_group)
ua_test_scale <- mutate(ua_test_scale,rating_binary=as.numeric(rating>=rating_mean))
ua_test_scale$rating_binary_3 <- 0
index_gt_3 <- which(ua_test_scale$rating>=3)
ua_test_scale$rating_binary_3[index_gt_3] <- 1
path_binary_test <- paste(path_binary,'ua_test_df_continuous_binary.txt')
#write.table(ua_test_scale,file=path_binary_test,row.names=FALSE)

##定义将dataframe转化成matrix的函数
df_to_mat <- function(df){
  mat <- dcast(df,user_id~item_id)
  uid <- mat$user_id
  iid <- colnames(mat)
  mat <- as.matrix(mat[,-1])
  rownames(mat) <- uid
  colnames(mat) <- iid[-1]
  return(mat)
}

##将训练集转化为矩阵的形式:69878*10667,user*item
mat_train <- df_to_mat(ua.base)
##对于test集手动添加不在train集中的item项
item_add <- setdiff(colnames(mat_train),ua.test$item_id)
ua.test_fill <- rbind(ua.test[,1:3],data.frame(user_id=1,item_id=item_add,rating=NA))
ua.test_fill <- mutate(ua.test_fill,item_id=as.numeric(item_id))
mat_test <- df_to_mat(ua.test_fill)
##直接用RData把两个Matrix进行保存
path_RData <- paste(path_output,'matrix_original.RData',sep='')
# save("mat_test","mat_train",file=path_RData)
load(path_RData)

##基于训练集：求出每个用户的平均评分再进行二元化(标准化)处理
user_rating_mean <- apply(mat_train,1,mean,na.rm=T)
##如果用户的评分大于等于其平均分，则为好评
mat_to_binary <- function(mat){
  mat_scale <- mat-user_rating_mean
  mat_scale[which(mat_scale>=0)] <- 1
  mat_scale[which(mat_scale<0)] <- 0
  mat_scale[which(is.na(mat_scale)==T)] <- -1
  return(mat_scale)
}
mat_train_scale <- mat_to_binary(mat_train)
mat_test_scale <- mat_to_binary(mat_test)
##用RData把Binary化之后的评分矩阵保存下来
path_RData <- paste(path_binary,'matrix_binary.RData',sep='')
# save("mat_test_scale","mat_train_scale",file=path_RData)
load(path_RData)

##评分>=3的赋值1，否则为0，进行二元化处理
##如果用户的评分大于等于3分，则为好评
mat_to_binary_3 <- function(mat){
  mat_scale <- mat
  mat_scale[which(mat>=3)] <- 1
  mat_scale[which(mat<3)] <- 0
  mat_scale[which(is.na(mat)==T)] <- -1
  return(mat_scale)
}
mat_train_scale_3 <- mat_to_binary_3(mat_train)
mat_test_scale_3 <- mat_to_binary_3(mat_test)
##用RData把Binary化之后的评分矩阵保存下来
path_RData <- paste(path_binary,'matrix_binary_3.RData',sep='')
# save("mat_test_scale_3","mat_train_scale_3",file=path_RData)
load(path_RData)

##没有近邻用户的添补计划
##实值的情况：以各个物品的平均打分进行填补
##输出各个物品的平均打分矩阵
vec_rating_mean <- apply(mat_train,2,mean,na.rm=TRUE)
value_mean <- mean(mat_train,na.rm=TRUE)
##如果平均分为NA，即该物品没人打分，那么使用全部物品的平均打分
index_na <- which(is.na(vec_rating_mean)==TRUE)
vec_rating_mean[index_na] <- value_mean
path_RData <- '/Users/baifrank/Desktop/recomm_output/MovieLens_10M_output/item_rating_mean.RData'
#save(vec_rating_mean,file=path_RData)
load(path_RData)

##计算各个物品的最大倾向性打分：以大多数人的打分情况为标准
##计算每一件物品中打分为1/0的人数
mat_train_scale_1 <- matrix(0,nrow(mat_train_scale),ncol(mat_train_scale))
mat_train_scale_1[which(mat_train_scale == 1)] <- 1
cnt_rating_1 <- apply(mat_train_scale_1,2,sum)
mat_train_scale_0 <- matrix(0,nrow(mat_train_scale),ncol(mat_train_scale))
mat_train_scale_0[which(mat_train_scale == 0)] <- 1
cnt_rating_0 <- apply(mat_train_scale_0,2,sum)
##得到大多数人对于各个物品的打分情况向量
vec_rating_mode <- as.numeric(cnt_rating_1 > cnt_rating_0)
path_RData <- '/Users/baifrank/Desktop/recomm_output/MovieLens_10M_output/data_binary/item_rating_mode.RData'
#save(vec_rating_mode,file=path_RData)
load(path_RData)

##gt3
mat_train_scale_3_1 <- matrix(0,nrow(mat_train_scale_3),ncol(mat_train_scale_3))
mat_train_scale_3_1[which(mat_train_scale_3 == 1)] <- 1
cnt_rating_1 <- apply(mat_train_scale_3_1,2,sum)
mat_train_scale_3_0 <- matrix(0,nrow(mat_train_scale_3),ncol(mat_train_scale_3))
mat_train_scale_3_0[which(mat_train_scale_3 == 0)] <- 1
cnt_rating_0 <- apply(mat_train_scale_3_0,2,sum)
##得到大多数人对于各个物品的打分情况向量
vec_rating_mode <- as.numeric(cnt_rating_1 > cnt_rating_0)
path_RData <- '/Users/baifrank/Desktop/recomm_output/MovieLens_10M_output/data_binary/item_rating_mode_3.RData'
#save(vec_rating_mode,file=path_RData)
load(path_RData)
