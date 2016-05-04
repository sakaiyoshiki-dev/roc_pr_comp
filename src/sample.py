#%matplotlib inline
import roc_pr_comp

if __name__ == '__main__':
    y_true = [1,0,1,1,0,0,0]
    y_predict = [7,6,5,4,3,2,1]
    roc_pr_comp.print_roc(y_true,y_predict)
    roc_pr_comp.print_precision_recall(y_true,y_predict)

    y_predict_mis = [6,7,5,4,3,2,1]
    roc_pr_comp.print_roc(y_true,y_predict_mis)
    roc_pr_comp.print_precision_recall(y_true,y_predict_mis)
