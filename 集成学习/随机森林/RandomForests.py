'''
随机森林最重要的调参数feature的个数选择
'''
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

iris = load_iris()
df = pd.DataFrame(iris.data,columns=iris.feature_names)

df['is_train'] = np.random.uniform(0,1,len(df))<=0.75
df['species'] = pd.Categorical.from_codes(iris.target,iris.target_names)

train,test = df[df.is_train==True],df[df.is_train==False]
features = df.columns[:4]
clf = RandomForestClassifier(n_jobs=2,max_features=4)
##标称型数据映射称为一组数字,返回值为一个元组，第一个是array，第二个是所有标称型元素
y,_ = pd.factorize(train['species'])
clf.fit(train[features],y)
preds = iris.target_names[clf.predict(test[features])]

print(pd.crosstab(test['species'], preds, rownames=['actual'], colnames=['preds']))
