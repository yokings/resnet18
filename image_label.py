import os
path='C:\\Users\wangcheng\Desktop\\test\\boxing'
a=os.listdir(path)
file=open('image_label.txt','w')
# print(a)
for j in range(len(a)):
	for i in os.listdir(path+'\\'+a[j]):
		b=path+'\\'+a[j]+'\\'+i
		print(b)
		file.write(b+' '+str(j)+'\n')
file.close

