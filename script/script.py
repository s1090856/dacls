import itertools as it
import os
base="es=early(1000,1000)\nteacher=TCN(normalized=False)\nstudent=TCN(normalized=True)\nlog=train_model([{}],epochs={},test_ratio={},drop_probability={},with_teacher={},batch_size={})\n"
houses=["dh1","rh1","dh1,dh2","rh1,rh3","dh1,rh1,dh2,rh3"]
epochs=["200"]
ratio=["0.2"]
drop=[str(x) for x in [0.0,0.3]]
teacher=[str(x) for x in [True,False]]
batch=[str(x) for x in [256,1024]]
count=0
comb="Set-{}_Epochs-{}_Ratio-{}_Drop-{}_Teacher-{}_BSize-{}"
csv="log.to_csv('{}.csv')\n"
latex="log.to_latex('{}.latex')\n"
figures=['Train_Loss','Test_Loss','Test_Accuracy','F1_macro','ANE']
plot="f=log.plot(y='{}',x='Epoch').get_figure()\nf.savefig('{}.png')\n"
for c in it.product(*[houses,epochs,ratio,drop,teacher,batch]):
	with open('template.py',mode='r') as I:
		lines=I.readlines()
		os.makedirs(comb.format(*c))
		with open( comb.format(*c)+"/"+comb.format(*c)+'.py',mode='w+') as O:
			O.writelines(lines)
			O.write(base.format(*c))
			O.write(csv.format(comb.format(*c)))
			O.write(latex.format(comb.format(*c)))
			for f in figures:
				O.write(plot.format(f,comb.format(*c)+"__"+f))
			print(base.format(*c))
	count+=1
	print(count)
