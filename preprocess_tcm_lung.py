import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

data_file = 'raw_data/TCM_Lung.xlsx'
tcm_lung = pd.read_excel(data_file)
# 样本数量
data_amount = tcm_lung.shape[0]

# 获取symptom syndrome treat herb数量
sym_amount = pd.read_excel(data_file, sheet_name='Symptom Dictionary').shape[0]
syn_amount = pd.read_excel(data_file, sheet_name='Syndrome Dictionary').shape[0]
treat_amount = pd.read_excel(data_file, sheet_name='Treat Dictionary').shape[0]
herb_amount = pd.read_excel(data_file, sheet_name='Herb Dictionary').shape[0]

# TCM_Lung数据集没有病人信息 我们假设每行是一个病人的信息
patients = []
# 构建列名
sym_col_names = [('Symptom' + str(i)) for i in range(1, sym_amount + 1)]
syn_col_names = [('Syndrome' + str(i)) for i in range(1, syn_amount + 1)]
treat_col_names = [('Treat' + str(i)) for i in range(1, treat_amount + 1)]
herb_col_names = [('Herb' + str(i)) for i in range(1, herb_amount + 1)]
all_col_names = sym_col_names + syn_col_names + treat_col_names + herb_col_names

row_cache = []
# 遍历原数据 每个样本进行转换
for i in range(data_amount):
    patient_visit = []
    # symptom
    sym_indexes = list(map(int, tcm_lung.iloc[i, 0].split(',')))
    sym_list = [1 if j in sym_indexes else 0 for j in range(1, sym_amount + 1)]

    # syndrom
    syn_indexes = list(map(int, tcm_lung.iloc[i, 1].split(',')))
    syn_list = [1 if j in syn_indexes else 0 for j in range(1, syn_amount + 1)]

    # treat
    treat_indexes = list(map(int, tcm_lung.iloc[i, 2].split(',')))
    treat_list = [1 if i in treat_indexes else 0 for i in range(1, treat_amount + 1)]

    # herb
    herb_indexes = list(map(int, tcm_lung.iloc[i, 3].split(',')))
    herb_list = [1 if i in herb_indexes else 0 for i in range(1, herb_amount + 1)]

    sample_data = sym_list + syn_list + treat_list + herb_list
    df = pd.Series(sample_data, index=all_col_names)
    # row_cache 为了之后构建完整的DataFrame
    row_cache.append(df.to_frame().T)
    visit = [df[sym_col_names], df[syn_col_names], df[treat_col_names], df[herb_col_names]]
    patient_visit.append(visit)
    patients.append(list(patient_visit))

# 所有数据的DataFrame
df_total = pd.concat(row_cache, ignore_index=True)
continuous_data = patients

X = df_total[sym_col_names]
Y1 = df_total[syn_col_names]
Y2 = df_total[treat_col_names]
Y3 = df_total[herb_col_names]
final_data = [X, Y1, Y2, Y3]
discrete_data = final_data

SEED = 42
# 分割数据集
train_data, test_data = train_test_split(continuous_data, test_size=0.2, random_state=SEED)

# 无序列的
pickle.dump(discrete_data, open('processed_data/processed_tcm_lung_discrete.pkl', 'wb'))
# 有序列的
pickle.dump([train_data, test_data], open('processed_data/processed_tcm_lung_continuous.pkl', 'wb'))

print('Preprocessing complete.')