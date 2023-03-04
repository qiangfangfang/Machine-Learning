from sklearn.metrics import log_loss
y_true = [[0,1],[0,1],[1,0],[1,0]]
y_pred1 = [[0.1,0.9],[0.2,0.8],[0.8,0.2],[0.49,0.51]]
y_pred2 = [[0.49,0.51],[0.45,0.55],[0.51,0.49],[0.1,0.9]]
print(log_loss(y_true,y_pred1))
print(log_loss(y_true,y_pred2))
