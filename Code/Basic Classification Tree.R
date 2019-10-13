# R-CODE FOR TREE

# tree
library(tree)
set.seed(2018)
reg.tree <- tree(train_label$x ~ ., train_data)
summary(reg.tree)
reg.tree
plot(reg.tree)
text(reg.tree)
pred <- predict(reg.tree, newdata = test_data)
pred = pred > 0.5
mean(pred!=test_label$x)


# pruned tree
set.seed(2018)
cv.reg.tree <- cv.tree(reg.tree)
plot(cv.reg.tree$size, cv.reg.tree$dev, type='b')
(min.cv.reg <- cv.reg.tree$size[which.min(cv.reg.tree$dev)])
# ANSWER: Optimal level of tree complexity is 6
prune.reg.tree <- prune.tree(reg.tree, best = min.cv.reg)
prune.pred <- predict(prune.reg.tree, newdata = test_data)
prune.pred = prune.pred > 0.5
mean(prune.pred!=test_label$x)
