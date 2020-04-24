# plotting lib
library(ggplot2)

# load data (annhui region)
respiration <- read.csv("anhui.csv")
rownames(respiration) = respiration$X

# plot
pl1 <- ggplot(data=respiration, aes(x=days,y=cases)) + geom_point( color="#993399", size=3)
pl1 + geom_line(color="#993399")

# model
Null.mod <- nls(cases~M/(1+exp(-beta*(days-alpha))), data=respiration, start=list(M=993,beta=0.28,alpha=13.3))
new.df <- data.frame(time=respiration$days)

# evaluate
new.df$pred1 <- predict(Null.mod, newdata=new.df)
r2 = cor(respiration$cases,new.df$pred1)^2
pl1 +  geom_line(data=new.df, aes(x=time,y=pred1), colour="#339900", size=1) + ggtitle(paste0("r2 = ",r2))
ggsave("anhui_model_vs_actual.png")

# now do the same for all the data
all_data = read.csv("all_china_data.csv",header = T)
all_data$days = seq(1,nrow(all_data),1) # add day


params.est <- data.frame()
regions = colnames(all_data[,2:34])
for (region in regions) {
  subset = cbind(all_data[,'days'],all_data[,region])
  colnames(subset) = c('days','cases')
  subset = as.data.frame(subset)
  Null.mod <- try(nls(cases~M/(1+exp(-beta*(days-alpha))), 
                  data=subset, 
                  start=list(M=993,beta=0.28,alpha=13.3)))
  if(inherits(Null.mod,"try-error")){
    print(region)
    next
  }
  new.df <- data.frame(days=all_data$days)
  new.df$pred1 <- predict(Null.mod, newdata=new.df)
  r2 = (cor(new.df$pred1,subset$cases))^2
  res = c(region,unname(coef(Null.mod)),r2)
  params.est <- rbind(params.est,as.data.frame(t(res)))
  
}
names(params.est) <- c("M","beta","alpha")
params.est


