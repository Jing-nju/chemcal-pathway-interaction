# -*- coding: UTF-8 -*- 
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import comb, ndtr
from scipy.stats import genextreme
from scipy.stats import norm
import scipy.io as io
from sklearn.metrics import auc

class MatrixBase():
    def __init__(self, row_type, row_id, col_type, col_id, mat):
        self.row_id = row_id
        self.row_type = row_type
        self.col_id = col_id
        self.col_type = col_type
        self.mat = mat
    
    def __repr__(self):
        return """{}:\n{}, \n{}:\n{},\nMatrix:\n{}
        """.format(self.row_type, self.row_id, self.col_type, self.col_id, self.mat)

    def get_value(self, row_id, col_id):
        assert row_id in self.row_id, "{} not in the row.".format(row_id)
        assert col_id in self.col_id, "{} not in the row.".format(col_id)
        idx_r = self.row_id.index(row_id)
        idx_c = self.col_id.index(col_id)
        return self.mat[idx_r, idx_c]
    
    def set_value(self, row_id, col_id, value):
        assert row_id in self.row_id, "{} not in the row.".format(row_id)
        assert col_id in self.col_id, "{} not in the row.".format(col_id)
        idx_r = self.row_id.index(row_id)
        idx_c = self.col_id.index(col_id)
        self.mat[idx_r, idx_c] = value

    def get_row(self, row_id):
        assert row_id in self.row_id, "{} not in the row.".format(row_id)
        idx_r = self.row_id.index(row_id)
        return self.mat[idx_r, :]

    def set_row(self, row_id, value):
        assert row_id in self.row_id, "{} not in the row.".format(row_id)
        assert self.mat.shape[1] == value.shape[0], "The length of writing value is mismatched with the matrix."
        idx_r = self.row_id.index(row_id)
        self.mat[idx_r, :] = value
    
    def get_col(self, col_id):
        assert col_id in self.col_id, "{} not in the row.".format(col_id)
        idx_c = self.col_id.index(col_id)
        return self.mat[:, idx_c]

    def set_col(self, col_id, value):
        assert col_id in self.col_id, "{} not in the row.".format(col_id)
        assert self.mat.shape[0] == value.shape[0], "The length of writing value is mismatched with the matrix."
        idx_c = self.col_id.index(col_id)
        self.mat[:, idx_c] = value


def extract_MatA(file):
    """
        Input: A file including multiple pathway-chemical info pairs.
        Output: A MatrixBase instance with a pathway-chemical 0/1 matrix and its indexes.
        ===
        Example: [TODO]
    """
    pathway_list, chemical_list = [], []
    edge_list = []

    with open(file, "r") as fin:
        for line in fin:
            pathway, chemical = line.split('\t')[:2]
            # Exclude the "\n" sign
            if '\n' in chemical:
                chemical = chemical[:-1] 
            
            if pathway not in pathway_list:
                pathway_list.append(pathway)
            if chemical not in chemical_list:
                chemical_list.append(chemical)
            edge_list.append((pathway, chemical))

    # Optional Sort Here
    pathway_list = sorted(pathway_list)
    chemical_list = sorted(chemical_list)
    
    num_pathway = len(pathway_list)
    num_chemical = len(chemical_list)

    matA = np.zeros((num_pathway, num_chemical))
    matA = MatrixBase("Pathway", pathway_list, "Chemical", chemical_list, matA)

    for edge in edge_list:
        matA.set_value(*edge, 1)
        # idx_pathway = pathway_list.index(edge[0])
        # idx_chemical = chemical_list.index(edge[1])
        # matA[idx_pathway, idx_chemical] = 1
    
    return matA

def extract_MatB(file):
    """
        Input: A file including multiple chemical-fingerprint info pairs.
        Output: A MatrixBase instance with a chemical-fingerprint matrix and its indexes.
        ===
        Example: [TODO]
    """
    chemical_list, fingerprint_list = [], []
    matB_list = list()
    with open(file, "r") as fin:
        fingerprint_list = fin.readline().split(',')
        del fingerprint_list[0]
        if '\n' in fingerprint_list[-1]:
            fingerprint_list[-1] = fingerprint_list[-1][:-1]
        # print(fingerprint_list)

        for line in fin:
            # print(line)
            candidate_list = line.split(',')
            if '\n' in fingerprint_list[-1]:
                candidate_list[-1] = candidate_list[-1][:-1]
            # print(candidate_list)
            chemical_list.append(candidate_list[0])
            elements = [int(candidate_list[x]) for x in range(1, len(candidate_list))]
            # print(elements)
            matB_list.append(elements)

    matB = np.array(matB_list)
    return MatrixBase("Chemical", chemical_list, "Fingerprint", fingerprint_list, matB)


def construct_matPv(pw_ch_mat, ch_fp_mat):
    pathways = pw_ch_mat.row_id
    chemicals = pw_ch_mat.col_id
    fingerprints = ch_fp_mat.col_id

    M = len(ch_fp_mat.row_id)
    K = np.sum(ch_fp_mat.mat > 0, axis=0)
    print(">>> K of fingerprints:", K)
    N = np.sum(pw_ch_mat.mat > 0, axis=1)
    print(">>> N of pathways:", N)

    Pv_list = []

    for idx_pw, pathway in enumerate(pathways):
        print(">>> Current calculating the Pv list of {}".format(pathway))
        # print(pw_ch_mat.mat[idx_pw])
        chem_idx = np.where(pw_ch_mat.mat[idx_pw] > 0)[0]
        # print(chem_idx)
        # print([chemicals[x] for x in chem_idx])
        chemicals_in_curr_pathway = [chemicals[x] for x in chem_idx]
        # X = len(chem_idx)
        pathway_Pv_list = []
        C_M_N = comb(M, N[idx_pw])
        for idx_fp, fingerprint in enumerate(fingerprints):
            X = sum([ch_fp_mat.get_value(_, fingerprint) > 0 for _ in chemicals_in_curr_pathway])
            print("fingerprint:{}\t(M, N, K, X) = ({}, {}, {}, {})".format(fingerprint, M, N[idx_pw], K[idx_fp], X))
            curr_Pv = 0
            for i in range(X, N[idx_pw]+1):
                C_K_i = comb(K[idx_fp], i)
                C_MsubK_Nsubi = comb(M - K[idx_fp], N[idx_pw] - i)
                curr_Pv += C_K_i * C_MsubK_Nsubi / C_M_N
            # print(curr_Pv)
            pathway_Pv_list.append(curr_Pv)
        Pv_list.append(pathway_Pv_list)
    matPv = np.array(Pv_list)
    return MatrixBase("Pathway", pathways, "Fingerprint", fingerprints, matPv)


def construct_matSc(chemical_x, pw_fp_mat, eps = 1e-10):
    chemical_list = []
    pathways = pw_fp_mat.row_id
    Sc_list = []
    for chemical in chemical_x:
        chemical_list.append(chemical)
        X = chemical_x[chemical]
        chemical_Sc_list = []
        for pathway in pathways:
            curr_Pv = pw_fp_mat.get_row(pathway)
            # print("Curr_Pv:", curr_Pv)
            # print("Curr_X:", X)
            # Example Test
            # curr_Sc = -np.sum(np.log10([0.52, 0.33, 0.21, 0.71, 0.28, 0.09]) * X)/np.sum(X)
            # Invalid value may encounter Here. Solved by add eps. When (np.sum(X) == 0)
            curr_Sc = -np.sum(np.log10(curr_Pv) * X) / (np.sum(X) + eps)
            # curr_Sc = np.log10(curr_Pv) * np.array(X)
            print("Chemical Name: {}\nPathway: {}\nSc Value: {:.2f}".format(chemical, pathway,curr_Sc))
            chemical_Sc_list.append(curr_Sc)
        Sc_list.append(chemical_Sc_list)

    matSc = np.array(Sc_list)
    return MatrixBase("Chemical_X", chemical_list, "Pathway", pathways, matSc)


def generate_random_chemicals(rand_num, ch_fp_mat):
    rand_chem_dicts = {}
    fp_size = len(ch_fp_mat.col_id)
    
    for i in range(rand_num):
        curr_fps = np.random.randint(0, 2, size=(fp_size)) # 0 or 1
        rand_chem_dicts["chem_rand"+str(i)] = curr_fps
    
    return rand_chem_dicts


def construct_matGenEV(ch_pw_mat):
    pathways = ch_pw_mat.col_id
    param_list = ['mean', 'std', 'location', 'scale']
    matGenEV = []
    for idx_pw, curr_pw in enumerate(pathways):
        data2fit = ch_pw_mat.mat[:, idx_pw].flatten()
        # Why NaN may exist here?
        if not np.isfinite(data2fit).all():
            print(data2fit)
        _, location, scale = genextreme.fit(data2fit, f0=0)
        mean, std = np.mean(data2fit), np.std(data2fit)
        matGenEV.append([mean, std, location, scale])
    matGenEV = np.array(matGenEV)
    return MatrixBase("Pathway", pathways, "EVparam", param_list, matGenEV)


def construct_matZ(ch_pw_mat, pw_ev_mat):
    chemicals = ch_pw_mat.row_id
    pathways = ch_pw_mat.col_id
    eps = 1e-7
    ch_pw_mat_t = ch_pw_mat.mat.transpose()
    matZ_t = np.zeros_like(ch_pw_mat_t)
    for idx_pw, curr_pw in enumerate(ch_pw_mat_t):
        matZ_t[idx_pw, :] = (ch_pw_mat_t[idx_pw, :] - pw_ev_mat.mat[idx_pw, 0]) / (pw_ev_mat.mat[idx_pw, 1] + eps)
    matZ = matZ_t.transpose()

    return MatrixBase("Chemical_X", chemicals, "Pathway", pathways, matZ)


def construct_matP(ch_pw_mat):
    chemicals = ch_pw_mat.row_id
    pathways = ch_pw_mat.col_id
    # matP = norm.sf(abs(ch_pw_mat.mat)) * 2
    matP = (1 - ndtr(ch_pw_mat.mat)) / 2.0

    return MatrixBase("Chemical_X", chemicals, "Pathway", pathways, matP)


def construct_matZ_and_matP_rev(ch_pw_mat, pw_ev_mat, eps=1e-7):
    chemicals = ch_pw_mat.row_id
    pathways = ch_pw_mat.col_id
    param_list = ['mean', 'std', 'location', 'scale']
    matZ = MatrixBase("Chemical", chemicals, "Pathway", pathways, np.zeros_like(ch_pw_mat.mat))
    matP = MatrixBase("Chemical", chemicals, "Pathway", pathways, np.zeros_like(ch_pw_mat.mat))

    for curr_pw in pathways:
        mean, std, location, scale = pw_ev_mat.get_row(curr_pw)
        print(">>> Handling the Z score and P value of Pathway {}".format(curr_pw))
        for curr_ch in chemicals:
            Z_value = (ch_pw_mat.get_value(curr_ch, curr_pw) - mean) / (std + eps)
            # matZ[:, idx_pw] = (matZ.get_col(curr_pw) - pw_ev_mat.get_value()) / std
            P_value = genextreme.sf(Z_value, 0, location, scale)
            print("\tChemicals {}: Z score({:.2f}) and P value({:.2f})".format(curr_ch, Z_value, P_value))
            matZ.set_value(curr_ch, curr_pw, Z_value)
            matP.set_value(curr_ch, curr_pw, P_value)
    
    return matZ, matP


def construct_matSummary(matSc, matZ, matP):
    chemicals = matSc.row_id
    pathways = matSc.col_id
    len_ch, len_pw = len(chemicals), len(pathways)

    tmp_f = lambda x: x.mat.flatten()
    data_Sc, data_Z, data_P = tmp_f(matSc), tmp_f(matZ), tmp_f(matP)
    data_pw = []
    for idx in range(len_ch):
        data_pw = data_pw + pathways
    data_ch = []
    for idx in range(len_ch):
        data_ch = data_ch + [chemicals[idx]] * len_pw
    
    return np.vstack((data_ch, data_pw, data_Sc, data_Z, data_P)).transpose()

    
def summary_evaluation(df_summary, chemicals, pathways, ref_table, kw_list):
    """
    Note that the id of `ref_table` is transposed compared with `df_summary`.
    """
    # 1. Sort P scores in ascending order in all chemicals.
    len_ch, len_pw = len(chemicals), len(pathways)
    df_summary['Sc'] = df_summary['Sc'].astype(np.float64)
    df_summary['Z score'] = df_summary['Z score'].astype(np.float64)
    df_summary['P value'] = df_summary['P value'].astype(np.float64)
    sorted_summary = pd.DataFrame(columns=kw_list)
    for ch in chemicals:
        df_ch = df_summary[df_summary['chemical'] == ch].sort_values(axis=0, by=['P value'])                
        # print("Chemicals {}:".format(ch))
        # print(df_ch)
        # print(df_ch['P value'].dtype)
        sorted_summary = pd.concat([sorted_summary, df_ch], ignore_index=True)
        # print(sorted_summary)

    # print(sorted_summary['P value'].dtypes)

    # print(sorted_summary)
    idx_pointer = np.array([len(pathways) * step for step in range(len_ch)])
    # print(idx_pointer)
    # print(ref_table)
    curr_TP, curr_FP, curr_FN, curr_TN = 0, 0, np.sum(ref_table.mat), len_ch * len_pw - np.sum(ref_table.mat)
    # TP, FP, FN, TN = [], [], [], []
    TPR, FPR = [], []
    # hit_rate = []
    # pw_idx = []
    print(curr_FN, curr_FP)

    print('>>> Calculating AUC...')
    for idx in range(1, len_pw + 1):
        # print(idx_pointer)
        curr_correct_pred = 0
        for curr_idx in idx_pointer:
            upd_pred = sorted_summary.iloc[curr_idx]
            # print(upd_pred)
            try:
                flag = ref_table.get_value(upd_pred['pathway'], upd_pred['chemical'])
            except:
                flag = 0 # if not exist in ref_table
            curr_correct_pred += flag
        idx_pointer += 1
        # if TP:
        #     curr_TP = TP[-1] + curr_correct_pred
        # else:
        #     curr_TP = curr_correct_pred
        curr_TP += curr_correct_pred
        curr_FP = idx * len_ch - curr_TP
        curr_FN -= curr_correct_pred
        curr_TN -= (len_ch - curr_correct_pred)
        
        curr_TPR = curr_TP / (curr_TP + curr_FN)
        curr_FPR = curr_FP / (curr_FP + curr_TN)
        TPR.append(curr_TPR)
        FPR.append(curr_FPR)
        # TP.append(curr_TP)
        # hit_rate.append(TP[-1] / (len_ch * idx))
        # pw_idx.append(idx)
        # print(TP, hit_rate)
        # print('N = {}, correct_prediction = {}, total_prediction ={}, hit rate = {:.2f}'.format(idx, int(TP[-1]), (len_ch * idx), hit_rate[-1]))
        print('N = {}, curr_TP = {}, curr_FP ={}, curr_FN = {}, curr_TN = {}, curr_TPR = {:.2f}, curr_FPR = {:.2f}'.format(idx, curr_TP, curr_FP, curr_FN, curr_TN, curr_TPR, curr_FPR))

    fin_AUC = auc(FPR, TPR)
    print(">>> Final AUC: ", fin_AUC)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    # Line y = x
    ax.scatter(FPR, TPR, marker='o')
    ax.plot(FPR, TPR, linestyle='-')
    auxiliary_line = np.linspace(0, 1, 100)
    ax.plot(auxiliary_line, auxiliary_line, linestyle='--')
    bbox = {"facecolor": "yellow", "alpha": 0.5}
    ax.text(0.75, 0.25, "AUC={:.2f}".format(fin_AUC), bbox=bbox, fontsize=15)
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=15)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=15)
    ax.grid()
    fig.tight_layout()
    plt.savefig('ROC_Curve.png')

    # zero_bound = [0 for _ in range(len_pw)]
    # fig, ax = plt.subplots(1, 1, figsize=(15, 6))
    # ax.plot(pw_idx, hit_rate, marker='o')
    # ax.fill_between(pw_idx, hit_rate, zero_bound, alpha=0.2)
    # ax.xaxis.set_major_locator(plt.FixedLocator(pw_idx))
    # ax.xaxis.set_tick_params(rotation=0)
    # ax.set_ylim([0, 1])
    # ax.set_xlabel('Number of Pathways')
    # ax.set_ylabel('Hit Rate')
    # ax.set_title('ROC Curve')
    # ax.grid()
    # plt.savefig('ROC_Curve.png')
    # # plt.show()
    # AUC = ( np.sum(hit_rate) - 0.5 * (hit_rate[0] + hit_rate[-1]) ) / (len_pw - 1)
    # print(">>> Final AUC: ", AUC)


def predict_evaluation(df_summary, chemicals, pathways, kw_list, topk=5):
    # 1. Sort P scores in ascending order in all chemicals.
    len_ch, len_pw = len(chemicals), len(pathways)
    df_summary['Sc'] = df_summary['Sc'].astype(np.float64)
    df_summary['Z score'] = df_summary['Z score'].astype(np.float64)
    df_summary['P value'] = df_summary['P value'].astype(np.float64)
    sorted_summary = pd.DataFrame(columns=kw_list)

    topk_kw_list = ['chemicals']
    for i in range(topk):
        topk_kw_list.append('Top{}'.format(i))
    topk_summary = pd.DataFrame(columns=topk_kw_list)

    for ch in chemicals:
        df_ch = df_summary[df_summary['chemical'] == ch].sort_values(axis=0, by=['P value'])
        # for i in range(topk):
        #     print(df_ch[i]['pathway'])      
        print("Chemicals {}: {}".format(ch, df_ch))
        candidate = df_ch['pathway'][: topk].tolist()
        # print([ch] + candidate)
        # curr_ch = pd.DataFrame(, columns=topk_kw_list)
        # topk_summary = pd.concat([topk_summary, curr_ch], ignore_index=True)
        topk_summary.loc[len(topk_summary)] = [ch] + candidate

        # print(df_ch['P value'].dtype)
        # sorted_summary = pd.concat([sorted_summary, df_ch], ignore_index=True)
        # print(topk_summary)
    
    print("Shape of the Chemical Summary:", topk_summary.shape)
    pred_chemical_summary_table_file_name = "pred_chemical_summary.csv"
    # kw_list = ['chemical', 'pathway', 'Sc', 'Z score', 'P value']
    df_summary = pd.DataFrame(topk_summary, columns=topk_kw_list)
    df_summary.to_csv(pred_chemical_summary_table_file_name, sep=',')


def build_null_model(fileA, fileB, rand_num, flow_cfg):
    # 1. Extract the MatA and the MatB.
    matA = extract_MatA(fileA)
    # print(matA)

    matB = extract_MatB(fileB)
    # print(matB)

    # [Note] Ensure the matA and matB are correct.

    # 2. Calculate the Pv of every pathway.
    if flow_cfg['Pv_preload'] is False:
        print(">>> Calculate the Pv of every pathway.")
        matPv = construct_matPv(matA, matB)
        with open(flow_cfg['Pv_path'] + '.pkl', 'wb') as f:
            pickle.dump(matPv, f)
    else:
        print(">>> Restore the Pv of every pathway.")
        with open(flow_cfg['Pv_path'] + '.pkl', 'rb') as f:
            matPv = pickle.load(f)
    # io.savemat(flow_cfg['Pv_path'] + '.mat', {'matPv': matPv})

    # print(matPv)

    # 3. Calculate the Sc of chemical_x
    # test_vec_chemical_dict = {
    #     "chemical_X": [0, 1, 0, 0, 1, 1]
    # }
    # matSc = construct_matSc(test_vec_chemical_dict, matPv)
    # print(matSc)

    # 4. Construct a Null Model
    if flow_cfg['real_Sc_preload'] is False:
        print(">>> Calculate the real data Sc.")
        matB_binary = np.where(matB.mat > 0, 1, 0)
        pred_chemical_dict = dict(zip(matB.row_id, matB_binary))
        real_matSc = construct_matSc(pred_chemical_dict, matPv)
        # print(real_matSc)
        with open(flow_cfg['real_Sc_path'] + '.pkl', 'wb') as f:
            pickle.dump(real_matSc, f)
    else:
        print(">>> Restore the real data Sc.")
        with open(flow_cfg['real_Sc_path'] + '.pkl', 'rb') as f:
            real_matSc = pickle.load(f)
    # io.savemat(flow_cfg['real_Sc_path'] + '.mat', {'real_matSc': real_matSc})

    # for key, value in pred_chemical_dict.items():
    #     print("Key:{}\tValue:{}".format(key, value))
    if flow_cfg['rand_Sc_preload'] is False:
        print(">>> Calculate the random data Sc.")
        rand_chemical_dict = generate_random_chemicals(rand_num, matB)
        # print(rand_chemical_dict)
        # Using the step 3 to calculate the Sc of chemical_x
        rand_matSc = construct_matSc(rand_chemical_dict, matPv)
        with open(flow_cfg['rand_Sc_path'] + '.pkl', 'wb') as f:
            pickle.dump(rand_matSc, f)
    else:
        print(">>> Restore the random data Sc.")
        with open(flow_cfg['rand_Sc_path'] + '.pkl', 'rb') as f:
            rand_matSc = pickle.load(f)
    # io.savemat(flow_cfg['rand_Sc_path'] + '.mat', {'rand_matSc': rand_matSc})

    # # # For rand_matSc visualized analysis for extreme value distribution.
    # # plt.plot(rand_matSc.mat[:, 0])
    # # Plot the Sc-Pv matrix to verify the trend of extreme value distribution
    # fig, ax = plt.subplots(figsize=(16, 8))
    # for idx_pw, curr_pw in enumerate(rand_matSc.col_id):
    #     ax.hist(rand_matSc.mat[:, idx_pw], bins=30, density=True, alpha=0.5, histtype='stepfilled', edgecolor='none', label=curr_pw)
    # ax.legend(bbox_to_anchor=(0, -0.2), ncol=5, loc='upper center')
    # fig.tight_layout()
    # fig.savefig("data_extreme_distribution.png", bbox_inches='tight')
    # # plt.show()

    # Generalized Extreme Value Distribution (Type I)， using rand_matSc
    if flow_cfg['EV_preload'] is False:
        print(">>> Calculate the Generalized Extreme Value Distribution.")
        matGenEV = construct_matGenEV(rand_matSc)
        with open(flow_cfg['EV_path'] + '.pkl', 'wb') as f:
            pickle.dump(matGenEV, f)
    else:
        print(">>> Restore the Generalized Extreme Value Distribution.")
        with open(flow_cfg['EV_path'] + '.pkl', 'rb') as f:
            matGenEV = pickle.load(f)
    # io.savemat(flow_cfg['EV_path'] + '.mat', {'matGenEV': matGenEV})


    # print(matGenEV)
    # print(matGenEV.row_id)
    # print(matGenEV.mat)
    null_model_paras_file_name = "null_model_paras.csv"
    df_null_model = pd.DataFrame(matGenEV.mat, columns=['mean', 'std', 'location', 'scale'], index=matGenEV.row_id)
    df_null_model.to_csv(null_model_paras_file_name, sep=',')

    # [Note] Chao Fang: This process has been abondoned.
    #               And the revision the list below.
    # # Calculate Z Score, using real_matSc
    # real_matZ = construct_matZ(real_matSc, matGenEV) # rand_Sc or real_Sc?
    # print(real_matZ)
    # # Calulate P Value
    # real_matP = construct_matP(real_matZ)
    # print(real_matP)
    if flow_cfg['Zvalue_preload'] is False and flow_cfg['Pscore_preload'] is False:
        print(">>> Calculate the Z Value and P Score.")
        real_matZ, real_matP = construct_matZ_and_matP_rev(real_matSc, matGenEV)
        with open(flow_cfg['Zvalue_path'] + '.pkl', 'wb') as fz, open(flow_cfg['Pscore_path'] + '.pkl', 'wb') as fp:
            pickle.dump(real_matZ, fz)
            pickle.dump(real_matP, fp)
    else:
        print(">>> Restore the Z Value and P Score.")
        with open(flow_cfg['Zvalue_path'] + '.pkl', 'rb') as fz, open(flow_cfg['Pscore_path'] + '.pkl', 'rb') as fp:
            real_matZ = pickle.load(fz)
            real_matP = pickle.load(fp)
    # io.savemat(flow_cfg['Zvalue_path'] + '.mat', {'real_matZ': real_matZ})
    # io.savemat(flow_cfg['Pscore_path'] + '.mat', {'real_matP': real_matP})
    
    mat_summary = construct_matSummary(real_matSc, real_matZ, real_matP)
    print("Shape of the Chemical Summary:", mat_summary.shape)
    pred_chemical_summary_table_file_name = "real_chemical_summary.csv"
    kw_list = ['chemical', 'pathway', 'Sc', 'Z score', 'P value']
    df_summary = pd.DataFrame(mat_summary, columns=kw_list)
    df_summary.to_csv(pred_chemical_summary_table_file_name, sep=',')
    
    chemicals = real_matSc.row_id
    pathways = real_matSc.col_id
    summary_evaluation(df_summary, chemicals, pathways, matA, kw_list)
    

def predict_via_null_model(fileB, topk, flow_cfg):
    # 'Pv_path': 'pv_generated',
    # 'EV_path': 'ev_generated',
    # matA = extract_MatA(fileA)
    matB = extract_MatB(fileB)

    # 2. Calculate the Pv of every pathway.
    print(">>> Restore the Pv of every pathway.")
    with open(flow_cfg['Pv_path'] + '.pkl', 'rb') as f:
        matPv = pickle.load(f)

    print(">>> Calculate the test data Sc.")
    matB_binary = np.where(matB.mat > 0, 1, 0)
    pred_chemical_dict = dict(zip(matB.row_id, matB_binary))
    pred_matSc = construct_matSc(pred_chemical_dict, matPv)

    print(">>> Restore the Generalized Extreme Value Distribution.")
    with open(flow_cfg['EV_path'] + '.pkl', 'rb') as f:
        matGenEV = pickle.load(f)
    
    print(">>> Calculate the Z Value and P Score.")
    pred_matZ, pred_matP = construct_matZ_and_matP_rev(pred_matSc, matGenEV)

    mat_summary = construct_matSummary(pred_matSc, pred_matZ, pred_matP)
    print("Shape of the Chemical Summary:", mat_summary.shape)
    pred_chemical_summary_table_file_name = "pred_chemical_summary.csv"
    kw_list = ['chemical', 'pathway', 'Sc', 'Z score', 'P value']
    df_summary = pd.DataFrame(mat_summary, columns=kw_list)
    df_summary.to_csv(pred_chemical_summary_table_file_name, sep=',')
    
    chemicals = pred_matSc.row_id
    pathways = pred_matSc.col_id
    predict_evaluation(df_summary, chemicals, pathways, kw_list, topk)
    

if __name__ == '__main__':

    mode = 'build' # 'predict' or 'build'

    if mode is 'build':

        # Configurations.
        # Maybe it would be better if we set configurations here?
        rand_num = 100000
        # fileA = "matA_pathway-chemical.txt"
        # fileA = "sample_keggpath.tsv"
        fileA = "keggpath.tsv"
        # fileB = "matB_chemical-fingerprint.txt"
        fileB = "KEGG.csv"

        # 数据是否需要重新构建
        real_frozen = False # False # True
        # 是否恢复原先的随机数进程
        rand_resume = False # False # True

        flow_cfg = {
            'Pv_preload': real_frozen,
            'real_Sc_preload': real_frozen,
            'rand_Sc_preload': rand_resume,
            'EV_preload': rand_resume,
            'Zvalue_preload': rand_resume & real_frozen,
            'Pscore_preload': rand_resume & real_frozen,

            'Pv_path': 'pv_generated',
            'real_Sc_path': 'real_sc_generated',
            'rand_Sc_path': 'rand_sc_generated',
            'EV_path': 'ev_generated',
            'Zvalue_path': 'zvalue_generated',
            'Pscore_path': 'pscore_generated'
        }

        build_null_model(fileA, fileB, rand_num, flow_cfg)

    elif mode is 'predict':

        # Configurations.
        fileB = "msdial neg.csv"
        # fileB = "keggpath_fingerprints.csv"
        topk = 5
        flow_cfg = {
            'Pv_path': 'pv_generated',
            # 'real_Sc_path': 'real_sc_generated',
            'EV_path': 'ev_generated',
            # 'Zvalue_path': 'zvalue_generated',
            # 'Pscore_path': 'pscore_generated'
        }

        predict_via_null_model(fileB, topk, flow_cfg)

    else:
        print("[Error] Unrecognized Mode Detected.")
