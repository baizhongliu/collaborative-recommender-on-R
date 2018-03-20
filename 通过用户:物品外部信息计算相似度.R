rm(list=ls())

library(data.table)
library(dplyr)
library(reshape2)
library(MASS)
library(dplyr)
u <- read.delim("~/Desktop/ml-100k/u.data", header=FALSE)
u.item <- fread("~/Desktop/ml-100k/u.item")##第一列是原始的movie_id
u.user <- fread("~/Desktop/ml-100k/u.user")##第一列是原始的user_id
##列名赋值
u.item_cols <- c("movie_id","movie_title","release_date","vedio_release_date",
               "IMDb_URL","unknown","Action","Adventure","Animation",
               "Children's","Comedy","Crime","Documentary","Drama","Fantasy",
               "Film-Noir","Horror","Musical","Mystery","Romance","Sci-Fi",
               "Thriller","War","Western")
colnames(u.item) <- u.item_cols
u.user_cols <- c("user_id","age","gender","occupation","zip code")
colnames(u.user) <- u.user_cols

##按照RBM输出矩阵的形式转化id
uid_transform <- fread("/Users/baifrank/Desktop/recomm_output/uid_transform.csv")
iid_transform <- fread("/Users/baifrank/Desktop/recomm_output/iid_transform.csv")

######################item
item_info <- left_join(u.item,iid_transform,by=c("movie_id"="iid"))
item_info <- arrange(item_info,iid_transform)
item_info <- item_info[,-c(1:6,25)]

nb_item <- nrow(item_info)
mat_sim_item <- matrix(0,nrow=nb_item,ncol=nb_item)

for(i in 1:(nb_item-1)){
  print(paste("Calculating",i,"/",(nb_item-1),sep=""))
  item1 <- item_info[i,]
  index1 <- which(item1==1)
  if(length(index1)==0){mat_sim_item[i,]<-0;next}
  for(j in (i+1):nb_item){
    #print(paste(i,j,sep='-'))
    item2 <- item_info[j,]
    index2 <- which(item2==1)
    if(length(index2)==0){mat_sim_item[i,j]<-0;next}
    index_intersect <- intersect(index1,index2)
    mat_sim_item[i,j] <- length(index_intersect)/min(length(index1),length(index2))
  }
}

######################user
user_info <- left_join(u.user,uid_transform,by=c("user_id"="uid"))
user_info <- arrange(user_info,uid_transform)
user_info <- user_info[,-c(1,6)]
##四个部分的相似度各为0.25（年龄之差在十岁之内当做相似）
nb_user <- nrow(user_info)
mat_sim_user <- matrix(0,nrow=nb_user,ncol=nb_user)
for(i in 1:(nb_user-1)){
  print(paste("Calculating",i,"/",(nb_user-1),sep=""))
  user1 <- user_info[i,]
  for(j in (i+1):nb_user){
    user2 <- user_info[j,]
    if(abs(user1$age-user2$age)<=10){sim_age <- 0.25}else{sim_age <- 0}
    if(user1$gender==user2$gender){sim_gender <- 0.25}else{sim_gender <- 0}
    if(user1$occupation==user2$occupation){sim_job <- 0.25}else{sim_job <- 0}
    if(user1$occupation=="other" || user2$occupation=="other"){sim_job <- 0}
    if(user1$`zip code`==user2$`zip code`){sim_zip <- 0.25}else{sim_zip <- 0}
    similarity <- sim_age+sim_gender+sim_job+sim_zip
    mat_sim_user[i,j] <- similarity
  }
}
##根据用户和物品的外部信息得到其相似性：mat_sim_item,mat_sim_user
##首先填满mat_sim
mat_sim_item_full <- mat_sim_item+t(mat_sim_item)
mat_sim_user_full <- mat_sim_user+t(mat_sim_user)

##user based：对于给定user和item,找到其最近邻用户的打分
##用户1对于物品1的打分：找到用户1打过分的物品，再找到与物品1相似的物品，得到其打分
##最终打分=打分*相似度
mat_rating_support <- matrix(0,nrow=nb_user,ncol=nb_item)
for(i in 1:nb_user){
  print(paste("Calculating",i,"/",nb_user,sep=""))
  user1_rating <- mat_train[i,]
  index_rating <- which(is.na(user1_rating)==FALSE)
  vec_rating <- user1_rating[index_rating]
  for(j in 1:nb_item){
    vec_sim <- mat_sim_item[j,index_rating]
    sim_max <- max(vec_sim)
    if(sim_max==0){mat_rating_support[i,j] <- 0;next}
    user1_rating_max <- vec_rating[which(vec_sim==sim_max)]
    mat_rating_support[i,j] <- mean(user1_rating_max*sim_max)
  }
}
mat_rating_support_u <- round(mat_rating_support)

##item based
mean_value <- mean(mat_train,na.rm=T)
mat_rating_support <- matrix(0,nrow=nb_item,ncol=nb_user)
for(i in 1:nb_item){
  print(paste("Calculating",i,"/",nb_item,sep=""))
  item1_rating <- t(mat_train)[i,]
  index_rating <- which(is.na(item1_rating)==FALSE)
  if(length(index_rating)==0){mat_rating_support[i,] <- mean_value;next}
  vec_rating <- item1_rating[index_rating]
  for(j in 1:nb_user){
    vec_sim <- mat_sim_user[j,index_rating]
    sim_max <- max(vec_sim)
    if(sim_max==0){mat_rating_support[i,j] <- 0;next}
    item1_rating_max <- vec_rating[which(vec_sim==sim_max)]
    mat_rating_support[i,j] <- mean(item1_rating_max*sim_max)
  }
}
mat_rating_support_i <- round(mat_rating_support)
##得到了根据user/item本身的相似度得到的相似user/item的辅助打分
#mat_rating_support_u,mat_rating_support_i

##读取RBM的最优打分，转化成matrix
rating_rbm <- fread("/Users/baifrank/Desktop/recomm_output/pre_best_100k.csv")
rating_rbm <- as.data.frame(rating_rbm)
colnames(rating_rbm) <- c('iid','value','uid')
mat_rbm <- dcast(rating_rbm,uid ~ iid)
mat_rbm <- as.matrix(mat_rbm[,-1])

##输入两个矩阵，计算误差的函数
compute_error <- function(mat_real,mat_pre){
  l_error <- list()
  mat_dif <- abs(mat_real-mat_pre)
  index_missing <- which(is.na(mat_real)==TRUE)
  mae <- mean(mat_dif[-index_missing])
  mse <- mean(mat_dif[-index_missing]^2)
  l_error$mae <- mae
  l_error$mse <- mse
  return(l_error)
}
##rbm误差
compute_error(mat_train,mat_rbm)
compute_error(mat_test,mat_rbm)
##support_u误差
compute_error(mat_train,mat_rating_support_u)
compute_error(mat_test,mat_rating_support_u)
##support_i误差
compute_error(mat_train,t(mat_rating_support_i))
compute_error(mat_test,t(mat_rating_support_i))

##构造一个包含rbm、support_u、support_i的dataframe
index_missing <- which(is.na(mat_train)==TRUE)
y=as.vector(mat_train[-index_missing])
data_train <- data.frame(y=y,
                         rbm=as.vector(mat_rbm[-index_missing]),
                         u=as.vector(mat_rating_support_u[-index_missing]),
                         i=as.vector(t(mat_rating_support_i)[-index_missing]))
##拟合一个regressor
lm_model <- lm(data=data_train,y~rbm+u+i)
y_pre <- predict(lm_model,newdata=data_train[,-1])
y_pre <- round(y_pre)
y_pre[which(y_pre>5)] <- 5
y_pre[which(y_pre<0)] <- 0
mean(abs(y-y_pre))
mean(abs(y-y_pre)^2)

index_test <- which(is.na(mat_test)==FALSE)
y <- as.vector(mat_test[index_test])
data_test <- data.frame(rbm=as.vector(mat_rbm[index_test]),
                        u=as.vector(mat_rating_support_u[index_test]),
                        i=as.vector(t(mat_rating_support_i)[index_test]))
y_pre <- predict(lm_model,newdata=data_test)
y_pre <- round(y_pre)
y_pre[which(y_pre>5)] <- 5
y_pre[which(y_pre<0)] <- 0
mean(abs(y-y_pre))##稍微有减小
mean(abs(y-y_pre)^2)
############################################################################

##计算每个物品的最流行打分
get_popular <- function(vec){
  index_na <- which(is.na(vec)==TRUE)
  if(length(index_na)==length(vec)){return(mean_value);next}
  vec_tb <- as.data.frame(table(vec[-index_na]))
  vec_tb1 <- filter(vec_tb,Freq==max(vec_tb$Freq))$Var1
  rt <- as.numeric(as.character(vec_tb1))
  return(max(rt))
}

rt_popular_u <- apply(mat_train,2,get_popular)
rt_popular_i <- apply(mat_train,1,get_popular)
##将向量转化成矩阵
mat_popular_i <- round(matrix(rep(rt_popular_i,1682),nrow=943))
mat_popular_u <- round(matrix(rep(rt_popular_u,each=943),nrow=943))

compute_error(mat_train,mat_popular_u)
compute_error(mat_test,mat_popular_u)

compute_error(mat_train,mat_popular_i)
compute_error(mat_test,mat_popular_i)

##构造一个包含rbm、popular_u、popular_i的dataframe
index_missing <- which(is.na(mat_train)==TRUE)
y=as.vector(mat_train[-index_missing])
data_train <- data.frame(y=y,
                         rbm=as.vector(mat_rbm[-index_missing]),
                         u=as.vector(mat_popular_u[-index_missing]),
                         i=as.vector(mat_popular_i[-index_missing]))
##拟合一个regressor
lm_model <- lm(data=data_train,y~rbm+u+i)
y_pre <- predict(lm_model,newdata=data_train[,-1])
y_pre <- round(y_pre)
y_pre[which(y_pre>5)] <- 5
y_pre[which(y_pre<0)] <- 0
mean(abs(y-y_pre))
mean(abs(y-y_pre)^2)

######
index_test <- which(is.na(mat_test)==FALSE)
y <- as.vector(mat_test[index_test])
data_test <- data.frame(y=y,rbm=as.vector(mat_rbm[index_test]),
                        u=as.vector(mat_popular_u[index_test]),
                        i=as.vector(mat_popular_i[index_test]))
lm_model <- lm(y~.,data_test)
summary(lm_model)
y_pre <- predict(lm_model,newdata=data_train[,-1])
y_pre <- round(y_pre)
y_pre[which(y_pre>5)] <- 5
y_pre[which(y_pre<0)] <- 0
mean(abs(y-y_pre))##稍微有减小
mean(abs(y-y_pre)^2)


