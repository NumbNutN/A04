### 代码规范

+ 供其他人使用的代码最好写好功用和参数、返回值等注释，参数和返回值类型最好显式的标注数据类型

+ 注释推荐采用VSCode插件"autoDocstring - Python Docstring Generator"进行模板生成

  

### 项目组织规范

+ 可以被共享的工具类方法可以放在git主目录/tool下，作为tool工具包的一部分
+ 其余代码可以存放在其他区域

您可以看一下当前git仓库的目录组织形式
```
├── classification_base_on_BOW.py
├── classification_base_on_CNN_&_vector.py
├── photo_gets.py
├── readme.md
├── data
│   ├── test(unlabeled).csv	 	(仓库未包含)
│   ├── train1补充.xlsx		   (仓库未包含)
│   ├── train1.csv				(仓库未包含)
│   ├── train2.csv				(仓库未包含)
│   └── train3.xlsx				(仓库未包含)
├── driver
│   └── chromedriver_win32
│       ├── chromedriver.exe
│       └── LICENSE.chromedriver
├── issue
│   ├── ISSUE.md
│   ├── RFC_230315_01.md
│   └── RFC_230316_01.md
├── log
│   └── log_2023_3_18.md
└── tool
    ├── classification_tool.py
    ├── evaluation_tool.py
    └── feature_extraction_tool.py
```

##### 我怎么使用Tool？

如果您使用了tool库中我们定义的工具（主要用于做特征工程、分类和结果评估等）

且您的文件放置的路径和tool文件夹不在同一路径下，可以参考以下方式

```
import sys
sys.path.append("依据相对地址访问tool文件夹所在路径")

'''
如您的代码组织是
├── new
│   └── your_new_file.py
└── tool
    ├── classification_tool.py
    ├── evaluation_tool.py
    └── feature_extraction_tool.py

则在your_new_file.py中sys.path.append("..") 添加上层目录（tool就在这）
'''
```



### 版本管理规范

+ 我们强烈建议你使用git clone在本地拉取一个和云端同步的库而不是选择把所有的文件拷贝下来。小项目代码变动大，您的代码很可能在几天后失效

+ 推荐使用Github Desktop git可视化管理界面，您可以非常直观的查看当前版本有什么新的代码变动，以及更快捷的代码commit  

  官网链接在这：[GitHub Desktop | Simple collaboration from your desktop](https://desktop.github.com/)

### 提交规范

+ 可以使用 pull request，也可以直接push
+ 分支不要求干净



### 用例介绍

+ classification_base_on_BOW.py

一个基于词袋模型特征模型和简单线性分类器的分类模型的Hello World
采集了train2.csv 婚恋交友 和 信贷理财 两个标签的 线性分类模型

+ photo_gets.py

selenium实现对网页快照的截取(图片形式)

### 项目依赖

python解释器 3.7.16


spacy 3.5.0
+ tool/classification_tool.py
+ tool/feature_extraction_tool.py
+ classification_base_on_BOW.py

numpy 1.21.6
+ tool/evaluation_tool.py
+ classification_base_on_BOW.py
  

selenium 4.8.2
+ photo_gets.py
  

scikit-learn 1.0.2

+ classification_base_on_BOW.py
+ tool/feature_extraction_tool.py

openpyxl 3.1.2
+ tool/classification_tool.py

pillow 9.4.0
+ photo_gets.py

pytesseract 0.3.10
+ photo_gets.py

tensorflow 2.11.0
+ TextCNN/cnn_model.py

keras 2.11.0

+ classification_base_on_CNN_&_vector.py

#### 模型
+ zh_core_web_md
```
nlp = spacy.load("zh_core_web_md)
```
+ zh_core_web_sm
```
nlp = spacy.load("zh_core_web_sm)
```

#### 如果您对搭建版本一致执行环境感到十分棘手，可以直接遵循以下指令保证开发环境的完全一致
```
# 使用Anaconda创建一个隔离的虚拟环境，指定python解释器版本
conda create -n YOUR_ENVS_NAME python=3.7
# 激活指定环境
conda activate YOUR_ENVS_NAME

#安装spacy和语言模型
#参考官网：https://spacy.io/usage
pip install -U pip setuptools wheel
pip install -U spacy
python -m spacy download zh_core_web_md

pip install scikit-learn
pip install keras
pip install tensorflow
```