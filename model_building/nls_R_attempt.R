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


