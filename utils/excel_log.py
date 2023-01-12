import pandas as pd
import os
from openpyxl import load_workbook

    
# global excel_path

def xlsx_init(log_path): 
        excel_path = log_path + "/fed_pacs.xlsx"
        
        Header = ['alg_mode', 'network', 'source', 'target', 'jig_weight', 
                  'total_iters', 'total_wk_iters', 'batch_size','a_iter',
                  'Train_Loss', 'Train_Class_Acc', 'Train_Jig_Acc',
                  'Val_Loss_L', 'Val_Class_Acc_L', 'Val_Jig_Acc_L',
                #   'Val_Loss_G', 'Val_Class_Acc_G', 'Val_Jig_Acc_G',
                  'Test_Loss', 'Test_Class_Acc', 'Test_Jig_Acc']
        if not os.path.exists(excel_path):
            df = pd.DataFrame(columns=Header)
            df.to_excel(excel_path, index=False)   

def xlsx_append(args, running_data, log_path):
        """ 
        refer to 
        https://blog.csdn.net/mygodit/article/details/97640770 
        https://zhuanlan.zhihu.com/p/340302599 
        """
        excel_path = log_path + "/fed_pacs.xlsx"
        
        content = [(args.alg_mode, args.network, args.source, args.target, args.jig_weight,
                   args.iters, args.wk_iters, args.batch, running_data['a_iter'],
                   running_data['train_loss'], running_data['train_acc_c'], running_data['train_acc_j'],
                   running_data['val_loss_l'], running_data['val_c_acc_l'], running_data['val_j_acc_l'],
                #    running_data['val_loss_g'], running_data['val_c_acc_g'], running_data['val_j_acc_g'],
                   running_data['test_loss'], running_data['test_c_acc'], running_data['test_j_acc'])]
        df = pd.DataFrame(content)
        # df_old = pd.DataFrame(pd.read_excel(excel_path))
        df_old = pd.read_excel(
                excel_path,
                engine='openpyxl',
                )
#         writer = pd.ExcelWriter(excel_path, mode='a')
        with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a') as writer:
        #     df.to_excel(writer)
            book=load_workbook(excel_path)
            writer.book = book
            writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
            df_rows = df_old.shape[0]
            df.to_excel(writer, startrow=df_rows+1, index=False, header=False) 
            writer.save()
