import ipdb

import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import warnings
warnings.filterwarnings("ignore")

model_path="./pretrained_model/descriptors/"
result_path="./viscosity_result/"

RF_name = 'RF_Opt.model'
KRR_name = 'KRR_Opt.model'
MLP_name = 'MLP_Opt.model'
XGboost_name = 'XGboost_Opt.model'

RF_model = joblib.load(model_path+RF_name)
KRR_model = joblib.load(model_path+KRR_name)
MLP_model = joblib.load(model_path+MLP_name)
XGboost_model = joblib.load(model_path+XGboost_name)

path="./dataset/"
file='Cor_train.pkl'
train_df= pd.read_pickle(path+file)

select_feature=['T', 'K_bond_ave', 'K_ang_ave', 'Theta0_ave', 'AATS1dv',
       'AATSC3dv', 'HybRatio', 'ETA_dBeta', 'IC1', 'nHRing', 'TSRW10']

train_select=train_df[select_feature]

dataset_path = "./dataset/Descriptor_data/313K/"

PolyInfo_file='PolyInfo_Des_313.pkl'
PolyInfo_df= pd.read_pickle(dataset_path+PolyInfo_file)
print(PolyInfo_df)
polyinfo_select=PolyInfo_df[select_feature]

PI_file='PI1M_Des_313.pkl'
PI_df= pd.read_pickle(dataset_path+PI_file)
PI_select=PI_df[select_feature]
print(PI_df)

Polyimides_file='Polyimides_Des_313.pkl'
Polyimides_df= pd.read_pickle(dataset_path+Polyimides_file)
print(Polyimides_df)
Polyimides_select=Polyimides_df[select_feature]


for i in [train_select,polyinfo_select,PI_select,Polyimides_select]:
    i.index = range(i.shape[0])
scaler.fit(train_select)
polyinfo_data=scaler.transform(polyinfo_select)
PI_data=scaler.transform(PI_select)
Polyimides_data=scaler.transform(Polyimides_select)

#PolyInfo database
polyinfo_RF = RF_model.predict(polyinfo_data)
polyinfo_KRR = KRR_model.predict(polyinfo_data)
polyinfo_MLP = MLP_model.predict(polyinfo_data)
polyinfo_XGboost = XGboost_model.predict(polyinfo_data)

#PI database
PI_RF = RF_model.predict(PI_data)
PI_KRR = KRR_model.predict(PI_data)
PI_MLP = MLP_model.predict(PI_data)
PI_XGboost = XGboost_model.predict(PI_data)

#Polyimides database
Polyimides_RF = RF_model.predict(Polyimides_data)
Polyimides_KRR = KRR_model.predict(Polyimides_data)
Polyimides_MLP = MLP_model.predict(Polyimides_data)
Polyimides_XGboost = XGboost_model.predict(Polyimides_data)

polyinfo_VI = pd.DataFrame({'SMILES':PolyInfo_df['SMILES'],'RF':polyinfo_RF,'KRR':polyinfo_KRR,'MLP':polyinfo_MLP,'XGboost':polyinfo_XGboost})
print(polyinfo_VI)
polyinfo_VI.to_csv(result_path + 'polyinfo_Viscosity_313.csv', index=False)

PI_VI = pd.DataFrame({'SMILES':PI_df['SMILES'],'RF':PI_RF,'KRR':PI_KRR,'MLP':PI_MLP,'XGboost':PI_XGboost})
print(PI_VI)
PI_VI.to_csv(result_path + 'PI_Viscosity_313.csv', index=False)

Polyimides_VI = pd.DataFrame({'SMILES':Polyimides_df['SMILES'],'RF':Polyimides_RF,'KRR':Polyimides_KRR,'MLP':Polyimides_MLP,'XGboost':Polyimides_XGboost})
print(Polyimides_VI)
Polyimides_VI.to_csv(result_path + 'Polyimides_Viscosity_313.csv', index=False)

#ipdb.set_trace() # ipdb debug break point


path = './viscosity_result/'    
df = pd.read_csv(path + 'screen_result.csv')  
    
filtered_df = df[(df['PVI'] > 500) & (df['SA'] < 2.9)& (df['TE'] < 32)]  
  
sorted_df = filtered_df.sort_values(by='PVI', ascending=False)  
  
sorted_df.to_csv(path + 'screen_result_pro.csv', index=False)  
  
print('=== Filtered and sorted data has been saved to ./viscosity_result/screen_result_pro.csv ===')



import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import AllChem
import os  
import glob 
from PIL import Image 

fig_start_id = 280
fig_end_id = 336 # [fig_start_id:fig_end_id]--[0:56] eg: [29:100]取第30个到第100个(不含101)
fig_range = str(fig_start_id) + '-' + str(fig_end_id)

image_dir = './viscosity_result/mol_fig/'  
#output_file = './viscosity_result/mol_fig/top112_mol_grid.png'  
#fig_start_id:fig_end_id
output_file = './viscosity_result/mol_fig/' + fig_range + '_mol_grid.png'  

# Find and delete all PNG files in this directory to prepare
#  for the correct stitching of subsequent images  
png_files = glob.glob(os.path.join(image_dir, '*.png'))  
  
for png_file in png_files:  
    try:  
        os.remove(png_file)  
        print(f"Deleted file: {png_file}")  
    except OSError as e:  
        print(f"Error: {e.strerror} : {png_file}")


data = pd.read_csv('./viscosity_result/screen_result.csv')

filtered_mol = data[(data['SA'] < 3.2) & (data['PVI'] > 480) & (data['TE'] < 30)]
filtered_mol.to_csv('./viscosity_result/screen_result_pro.csv', index=False)
filtered_mol_num = filtered_mol.shape[0]

print("----------------------------------------------------------------")
print("=============== SA < 3.2 & PVI > 480 & TE < 30 ===============")
print("The number of molecules that met the screening criteria was:")
print(f"===============[ {filtered_mol_num} ]===============")
print("== Filtered data set: ./viscosity_result/screen_result_pro.csv")
print("----------------------------------------------------------------")

filtered_mol_sorted = filtered_mol.sort_values(by='PVI', ascending=False)
filtered_mol_sorted.reset_index(drop=True, inplace=True)

#filtered_data = filtered_mol_sorted.head(112)   
#filtered_data = filtered_mol_sorted.iloc[0:4]
filtered_data = filtered_mol_sorted.iloc[fig_start_id:fig_end_id]

molecules = [Chem.MolFromSmiles(smiles) for smiles in filtered_data['SMILES']]

# Draw individual molecular pictures (non-grid pictures)
def draw_molecule(molecule, sa, pvi, te):
    mol = Chem.MolFromSmiles(molecule)
    d = rdMolDraw2D.MolDraw2DCairo(500, 220)  # 修改图像大小
    d.drawOptions().bondLineWidth = 1.0  # 调整键的线宽
    d.drawOptions().legendFontSize = 24  # 调整图例字体大小
    d.drawOptions().legendWidth = 200  # 调整图例宽度
    d.drawOptions().legendHeight = 100  # 调整图例高度
    d.drawOptions().atomLabelFontSize = 14  # 调整原子标签字体大小
    d.drawOptions().minFontSize = 21  # 元素字体大小
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol, legend='SA={:.2f}  PVI={:.0f}  TE={:.2f}'.format(sa, pvi, te))
    d.FinishDrawing()
    return d.GetDrawingText()

for index, row in filtered_data.iterrows():
    molecule = row['SMILES']
    sa = row['SA']
    pvi = row['PVI']
    te = row['TE']
    mol_image_text = draw_molecule(molecule, sa, pvi, te)
    
    with open(f'./viscosity_result/mol_fig/molecule_{index}.png', 'wb') as f:
        f.write(mol_image_text)


print("==== pictures stitching is runing...")
# The resulting molecular picture is spliced
images =[]
filenames = [] 

for filename in os.listdir(image_dir):  
    if filename.endswith('.png') and filename.startswith('molecule_'):  
        img = Image.open(os.path.join(image_dir, filename))  
        images.append(img)  
        filenames.append(filename)  
  
filenames.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))  

images_sorted = [images[filenames.index(f)] for f in filenames]  
 
images_per_row = 4  
total_rows = (len(filenames) + images_per_row - 1) // images_per_row  
  
total_width = 0  
max_height = 0  
for img in images_sorted[:images_per_row]:  
    total_width += img.width  
    max_height = max(max_height, img.height)  
  
#new_img = Image.new('RGB', (total_width * total_rows, max_height))  
new_img = Image.new('RGB', (total_width, max_height * total_rows))
    
y_offset = 0  
for img in images_sorted:  
    x_offset = (images.index(img) % images_per_row) * img.width  
    new_img.paste(img, (x_offset, y_offset))  
      
    if (images.index(img) + 1) % images_per_row == 0:  
        y_offset += max_height  
  
if (len(images) % images_per_row) != 0:  
    y_offset += max_height  
     
new_img.save(output_file)

for img in images:  
    img.close()

print(f"=======succeed!  ./mol_fig/{fig_range}_mol_grid.png=======")
