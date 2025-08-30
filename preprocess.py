import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# from transformers import BertTokenizer
import pickle
from collections import defaultdict
from sklearn.model_selection import train_test_split

SEED = 42
# 读取CSV文件
df = pd.read_csv('raw_data/data.csv', low_memory=False)
"""step1 选择所有需要的列"""
# 删除所有不需要的列
# df = df.drop(
#     columns=['姓名', 'visitTimes', '初复诊', '出生日期', 'visitDate', '填报人', '音色-少气', '音色-湿啰音', '音色-短气',
#              '音色-哮鸣', '音色-气喘', '血色-血色暗', '血色-血色红', '呼吸症状-气紧', '呼吸症状-喘累', '呼吸症状-胸闷',
#              '呼吸症状-气短',
#              ])

# 保留要的列
columns = [
    'patientUniqueId', '性别', '职业', '目', '面色', '面部浮肿', '神志', '体态',
    '形体', '舌质', '舌形', '舌苔', '苔色', '舌下脉络迂曲', '声音', '音色-少气', '音色-湿啰音',
    '音色-短气', '音色-哮鸣', '音色-气喘', '气味', '咳嗽', '咳痰', '咯血', '血色-血色暗',
    '血色-血色红', '胸痛', '呼吸', '呼吸症状-气紧', '呼吸症状-喘累', '呼吸症状-胸闷', '呼吸症状-气短',
    '恶寒', '出汗', '口味', '头身胸腹及不适-皮疹伴瘙痒', '头身胸腹及不适-周身沉重',
    '头身胸腹及不适-声音嘶哑', '头身胸腹及不适-舌麻', '头身胸腹及不适-眼痒', '头身胸腹及不适-周身疼痛',
    '头身胸腹及不适-心绞痛', '头身胸腹及不适-下肢酸软', '头身胸腹及不适-牙龈肿痛', '头身胸腹及不适-咽干',
    '头身胸腹及不适-腹痛', '头身胸腹及不适-手足麻木', '头身胸腹及不适-胃脘痛', '头身胸腹及不适-手足蜕皮',
    '头身胸腹及不适-清涕', '头身胸腹及不适-头晕', '头身胸腹及不适-口干', '头身胸腹及不适-恶心',
    '头身胸腹及不适-嗳气', '头身胸腹及不适-眼哆', '头身胸腹及不适-耳胀', '头身胸腹及不适-喷嚏',
    '头身胸腹及不适-耳聋', '头身胸腹及不适-痤疮', '头身胸腹及不适-胁胀', '头身胸腹及不适-舌痛',
    '头身胸腹及不适-干呕', '头身胸腹及不适-脱发', '头身胸腹及不适-腰痛', '头身胸腹及不适-心胸压榨感',
    '头身胸腹及不适-双下肢肿', '头身胸腹及不适-眼胀', '头身胸腹及不适-心慌', '头身胸腹及不适-反酸',
    '头身胸腹及不适-肩痛', '头身胸腹及不适-胁痛', '头身胸腹及不适-手足皲裂', '头身胸腹及不适-耳鸣',
    '头身胸腹及不适-周身乏力', '头身胸腹及不适-心悸', '头身胸腹及不适-口舌生疮', '头身胸腹及不适-呕吐',
    '头身胸腹及不适-脓涕', '头身胸腹及不适-呃逆', '头身胸腹及不适-头痛', '头身胸腹及不适-腹胀',
    '头身胸腹及不适-眼花', '头身胸腹及不适-咽痛', '头身胸腹及不适-鼻腔分泌物增多', '头身胸腹及不适-口渴',
    '头身胸腹及不适-眼雾', '头身胸腹及不适-眼痛', '头身胸腹及不适-口腔溃疡', '头身胸腹及不适-背痛',
    '头身胸腹及不适-鼻腔出血', '头身胸腹及不适-涎多', '头身胸腹及不适-胃脘胀', '头身胸腹及不适-咽痒',
    '精神', '饮食-食欲减退', '饮食-厌食', '饮食-厌油', '饮食-饥不欲食', '饮食-食欲可', '饮食-消谷善饥', '睡眠',
    '大便情况', '大便症状-排便不爽', '大便症状-肛门坠胀', '大便症状-肛门排气增多', '大便症状-便血',
    '大便症状-肛门灼热', '大便症状-里急后重', '小便情况', '尿量', '小便色', '小便症状-尿痛', '小便症状-尿急',
    '小便症状-遗尿', '小便症状-夜尿增多', '小便症状-小便失禁', '小便症状-排尿乏力', '小便症状-小便淋漓不尽',
    '体重', '脉象', '手术史', '手术方式', '手术方式其他补充', '肺切除方式', '肺切除方式其他补充', '淋巴结',
    '切除淋巴结数', '切缘肿瘤支气管', '切缘肿瘤脉管', '切缘肿瘤神经侵犯', '切缘肿瘤气道播散',
    '切缘肿瘤支气管切缘', '切缘肿瘤肿瘤附近组织', '化疗方案-奈达铂', '化疗方案-伊立替康', '化疗方案-培美曲塞',
    '化疗方案-紫杉醇', '化疗方案-长春瑞滨', '化疗方案-其他', '化疗方案-白蛋白紫杉醇', '化疗方案-卡铂',
    '化疗方案-顺铂', '化疗方案-依托泊苷', '化疗方案-多西他赛', '化疗方案-吉西他滨', '化疗方案其他补充',
    '化疗疗效评价', '靶向药', '靶向药其他补充', '免疫治疗', '免疫治疗其他补充', '免疫治疗疗效评价', '其他治疗史',
    '其他治疗史其他补充', '既往史-高血压', '既往史-冠心病', '既往史-糖尿病', '既往史-结核', '既往史-乙肝',
    '肺部基础病-哮喘', '肺部基础病-其他', '肺部基础病-慢阻肺', '肺部基础病-尘肺', '肺部基础病-支气管扩张',
    '肺部基础病-肺气肿', '个人史-饮酒', '个人史-二手烟吸入', '个人史-油烟吸入', '个人史-吸烟',
    '职业暴露史-放射性接触', '职业暴露史-其他', '职业暴露史-粉尘暴露', '月经量', '带下-色透明', '带下-色白',
    '带下-色黄', '家族史-肿瘤家族史', '病理检查', '免疫组化p40', '免疫组化p63',
    '免疫组化TTF-1', '免疫组化Napsin_A', '免疫组化SYN', '免疫组化CGA', '免疫组化CD56', '免疫组化CK5-6',
    '免疫组化CK-pan', '免疫组化Ki-67', '病理分级', '肺癌', '肺癌部位-右肺中叶', '肺癌部位-左肺下叶',
    '肺癌部位-左肺上叶', '肺癌部位-右肺下叶', '肺癌部位-右肺上叶', '肺癌病理类型', '肺癌病理类型其他补充',
    '肺癌CPR', '分期T', '分期N', '分期M', '分期', '西医诊断-胸腔积液', '西医诊断-癌性疼痛', '西医诊断-肺炎',
    '西医诊断-（淋巴结、胸膜、脑、骨、肝、肾上腺）继发恶性肿瘤', 
    '中医诊断-痰瘀互结证', '中医诊断-肝郁气滞','中医诊断-痰热蕴肺证', '中医诊断-脾虚毒蕴证',
    '处方']

df1 = df[columns]
text_columns = ['主诉', '持续性主诉']
continuous_columns = ['年龄', '有转移的淋巴结数']
# 获取离散列
discrete_columns = list(pd.Index(columns))
discrete_columns.remove('patientUniqueId')

"""step2 处理discrete列，离散列中的每个值都对应一个label"""
df_new = pd.DataFrame()
df_new['patientUniqueId'] = df1['patientUniqueId'].copy()
transformed_data = []
for col in discrete_columns:
    le = LabelEncoder()
    transformed_col = le.fit_transform(df1[col].astype(str))
    transformed_data.append(pd.DataFrame(transformed_col, columns=[col]))
# 使用pd.concat合并所有列，axis=1表示按列合并
df_new = pd.concat([df_new] + transformed_data, axis=1)

"""step3 提取几个y值"""
y1_columns = ['中医诊断-痰瘀互结证', '中医诊断-肝郁气滞', '中医诊断-痰热蕴肺证', '中医诊断-脾虚毒蕴证']
y2_columns = ['处方']
# 从df_new中获取列名，除去y_columns中的列和'patientUniqueId'
x_columns = [col for col in df_new.columns if col not in ['patientUniqueId'] + y1_columns + y2_columns]

"""之后提取所有药物"""
# 提取所有药物和对应的剂量
all_medicines = set()
for prescription in df["标准化处方"]:
    items = prescription.split('、')
    for item in items:
        # 提取药物和剂量，去掉末尾单位
        medicine = ''.join(filter(str.isalpha, item.rstrip('gmg')))
        dosage = ''.join(filter(lambda x: x.isdigit() or x == '.', item))
        all_medicines.add(medicine)

# 初始化结果DataFrame
result = pd.DataFrame(0, index=df.index, columns=list(all_medicines))

# 填充数据
for idx, prescription in enumerate(df["标准化处方"]):
    items = prescription.split('、')
    for item in items:
        # 提取药物和剂量，去掉末尾单位
        medicine = ''.join(filter(str.isalpha, item.rstrip('gmg')))
        dosage = ''.join(filter(lambda x: x.isdigit() or x == '.', item))
        result.loc[idx, medicine] = float(dosage) if dosage else 0

y3_columns = list(result.columns)

"""处理序列"""
# 1. 合并数据
df_combined = pd.concat([df_new, result], axis=1)

# 2. 将每个患者的数据组织成嵌套列表
grouped_data = df_combined.groupby('patientUniqueId')

# 创建最终的嵌套列表结构
patients = []
for patient_id, group in grouped_data:
    # 记录患者的所有就诊记录
    patient_visits = []
    for _, row in group.iterrows():
        # 每次就诊记录累积，形成新的样本
        visit = [row[x_columns], row[y1_columns], row[y2_columns], row[y3_columns]]
        patient_visits.append(visit)
        patients.append(list(patient_visits))  # 将当前患者的所有就诊记录累积成新的样本
continuous_data = patients

# # 输出结果
# print(patients)

X = df_new[x_columns]
Y1 = df_new[y1_columns]
Y2 = df_new[y2_columns]
Y3 = result
final_data = [X, Y1, Y2, Y3]
discrete_data = final_data

train_data, test_data = train_test_split(continuous_data, test_size=0.2, random_state=SEED)

# 无序列的
pickle.dump(discrete_data, open('processed_data/processed_data_discrete.pkl', 'wb'))
# 有序列的
pickle.dump([train_data, test_data], open('processed_data/processed_data_continuous.pkl', 'wb'))

print('Preprocessing complete.')
# 可以保存处理后的DataFrame或继续使用
# df.to_csv('processed_data.csv', index=False)
