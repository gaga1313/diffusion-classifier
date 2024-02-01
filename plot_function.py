import matplotlib.pyplot as plt


contrast_accuracy = {
'c100' : 0.917,
'c50' : 0.917,
'c30' : 0.917,
'c15' : 1.0,
'c10' : 0.667,
'c05' : 1.0,
'c03' : 0.5,
'c01' : 0.167,}

uniform_noise_accuracy = {'0.00' : 0.83,
'0.03' : 0.83,
'0.05' : 0.66,
'0.10 ': 0.67,
'0.20': 0.083,
'0.35' : 0.33,
'0.60 ': 0.0,
'0.90 ': 0.0}


plt.plot(list(uniform_noise_accuracy.keys()), list(uniform_noise_accuracy.values()), marker = 'o', color = 'y')
plt.title('Stable Diffusion Classifier v2.0')
plt.xlabel('noise levels')
plt.ylabel('classfication accuray')
plt.savefig('/media/data_cifs/projects/prj_model_vs_human/diffusion-classifier/diffusion-classifier/plots/uniform_noise.png')